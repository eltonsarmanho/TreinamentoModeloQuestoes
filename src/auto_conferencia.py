"""Experimento "gerar e conferir" (ciclo v8, 2026-10-23): o modelo resolve a questão às cegas e só vale se concordar.

Ideia para o app, sem API: depois de gerar uma questão, o mesmo modelo recebe só o enunciado e as cinco alternativas
(sem gabarito nem resolução) e escolhe uma letra. Se a letra difere do gabarito da questão, ela é descartada.
Aqui medimos, sem treino, se essa concordância separa as questões que o professor aprovou das que reprovou:
as 388 questões já rotuladas (`outputs/melhoria_v8/F/rotulados.jsonl`, critério núcleo).

  resolver   GPU/CPU local (llama-server). Retomável: grava `<saida>/respostas.jsonl` item a item.
  apurar     sem modelo: lê as respostas e imprime precisão, perdas e IC95 de Wilson.

Modos: `direto` (só a letra, ~1 s por questão), `cot` (raciocínio curto e "Resposta: X", ~11 s por questão no CPU) e
`critica` (2026-10-23, depois da validação no v8rftb: a `cot` deixou passar 18 inválidas, nenhuma por conta errada, e sim
por mais de uma alternativa correta, dados contraditórios ou insuficientes). A `critica` julga cada alternativa às cegas
e termina com CORRETAS / CONTRADICAO / SUFICIENTE. Só passa com exatamente uma correta (= gabarito), sem contradição e
com dados suficientes. Formato fora do esperado = descarte. `apurar --modo critica` também cruza com a `cot` gravada.
`--permutar` embaralha as alternativas (seed fixa) e mapeia a letra de volta, para a conferência não herdar o
viés de letra do gerador. Não escreve em data/ e não chama API.

Uso (o comando de treino/GPU é sempre do responsável):
  venv/bin/python src/auto_conferencia.py resolver --gguf outputs/qwen3_4b/Qwen3-4B-Q4_K_M.gguf --modo direto --permutar
  venv/bin/python src/auto_conferencia.py apurar --modo direto --permutar
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

ROTULADOS = ROOT / "outputs" / "melhoria_v8" / "F" / "rotulados.jsonl"
SAIDA = ROOT / "outputs" / "melhoria_v8" / "auto_conferencia"
SEED_PERMUTACAO = 20261123
LETRAS = "ABCDE"
SISTEMA = ("Você resolve questões de múltipla escolha de matemática do ensino fundamental. "
           "Se nenhuma alternativa estiver correta, escolha a que diz isso (por exemplo, 'Nenhuma das alternativas anteriores').")
FORMATO = {"direto": "Responda SOMENTE com a letra da alternativa correta (A, B, C, D ou E).",
           "cot": "Resolva com poucos passos e termine com uma última linha no formato: Resposta: X (X é a letra)."}
# v2 (2026-10-23): na v1, 4 de 5 respostas vieram só com as três linhas finais, sem conta (chute: Q1001 "A,D"
# para 70 - 45 = 25), e a Q1005 inverteu SIM/NAO. Agora a análise vem antes do veredito e SIM sempre indica problema.
VERSAO_CRITICA = "v2"
FORMATO["critica"] = (
    "Revise esta questão com rigor, sem supor que ela está certa. Escreva a análise ANTES do veredito:\n"
    "1. Resolva o problema passo a passo, mostrando as contas.\n"
    "2. Para cada alternativa, escreva uma linha 'X) correta' ou 'X) incorreta', com o motivo. A alternativa "
    "'Nenhuma das alternativas anteriores' só é correta se nenhuma outra for.\n"
    "3. Diga se algum dado do enunciado contradiz outro e se falta algum dado ou há mais de uma interpretação.\n"
    "Só depois da análise, termine com estas três linhas (SIM indica problema):\n"
    "CORRETAS: <letras das alternativas corretas, separadas por vírgula, ou NENHUMA>\n"
    "CONTRADICAO: <SIM se algum dado contradiz outro, senão NAO>\n"
    "FALTA_DADO: <SIM se falta dado ou há mais de uma interpretação, senão NAO>")
SISTEMA_CRITICA = ("Você é um professor de matemática do ensino fundamental que revisa questões de múltipla escolha "
                   "antes de uma prova. Aponte qualquer ambiguidade, dado contraditório ou dado que falta.")
MAX_TOKENS = {"direto": 8, "cot": 700, "critica": 1200}


def carregar_itens(caminho=ROTULADOS):
    """Itens rotulados únicos, com 5 alternativas e sem conflito de rótulo."""
    itens, vistos = [], set()
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        alt = r.get("alternativas") or {}
        if r.get("dup_de") or r.get("conflito_rotulo") or r["sig"] in vistos:
            continue
        if sorted(alt) != list(LETRAS) or r.get("gabarito_modelo") not in LETRAS:
            continue
        vistos.add(r["sig"])
        itens.append(r)
    return itens


def itens_de_folha(folha, chave):
    """Folha preenchida + chave (formato gerar_folha_professor / auditar_extras_rft) -> itens no formato de rotulados."""
    import csv
    k = json.loads(Path(chave).read_text(encoding="utf-8"))["itens"]
    nucleo = ("correcao_matematica", "resposta_unica", "dados_suficientes")
    itens = []
    with open(folha, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            c = k[r["id"]]
            itens.append({"uid": r["id"], "sig": r["id"], "enunciado": r["enunciado"],
                          "alternativas": {L: r[L] for L in LETRAS}, "gabarito_modelo": r["gabarito_modelo"],
                          "valida_nucleo": all(r[x].strip() == "s" for x in nucleo),
                          "status_verificador": c.get("status"), "braco": c.get("origem") or c.get("modelo"),
                          "ano": r["ano"]})
    return itens


def obter_itens(a):
    if a.folha:
        return itens_de_folha(a.folha, a.chave)
    return carregar_itens()


def sufixo(a):
    modo = a.modo + (VERSAO_CRITICA if a.modo == "critica" else "")
    return f"{modo}{'_perm' if a.permutar else ''}{'_' + a.rotulo if a.rotulo else ''}"


def ordem_alternativas(uid, permutar, alternativas=None):
    """Letra ORIGINAL exibida em cada posição. Identidade se não permutar. Uma alternativa do tipo
    "Nenhuma das ..." só faz sentido por último, então fica na última posição e o resto é embaralhado."""
    ordem = list(LETRAS)
    if permutar:
        nenhuma = [L for L in LETRAS if str((alternativas or {}).get(L, "")).strip().lower().startswith("nenhum")]
        livres = [L for L in LETRAS if L not in nenhuma]
        random.Random(f"{SEED_PERMUTACAO}:{uid}").shuffle(livres)
        ordem = livres + nenhuma
    return ordem


def montar_prompt(item, modo, permutar):
    ordem = ordem_alternativas(item["uid"], permutar, item["alternativas"])
    linhas = [f"{LETRAS[i]}) {item['alternativas'][orig]}" for i, orig in enumerate(ordem)]
    return f"{item['enunciado']}\n\n" + "\n".join(linhas) + f"\n\n{FORMATO[modo]}"


def extrair_letra(texto, modo, letra_nenhuma=None):
    """Letra EXIBIDA escolhida, ou None. `letra_nenhuma`: posição exibida da alternativa "Nenhuma das ...",
    para quando o resolvedor escreve a resposta por extenso ("Resposta: Nenhuma das alternativas anteriores")."""
    t = (texto or "").strip()
    if modo == "cot":
        m = re.findall(r"Resposta\s*\**\s*[:\-]?\s*\**\s*\(?([A-E])\b", t, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()
        if letra_nenhuma and re.search(r"Resposta\s*\**\s*[:\-]?\s*\**\s*Nenhuma", t, flags=re.IGNORECASE):
            return letra_nenhuma
        return None
    m = re.match(r"^\W*([A-E])\b", t, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _campo(texto, nome):
    m = re.findall(rf"{nome}\s*\**\s*:\s*\**\s*([^\n]*)", texto or "", flags=re.IGNORECASE)
    return m[-1].strip() if m else None


def extrair_critica(texto):
    """{"corretas": [letras exibidas], "contradicao": bool, "suficiente": bool} ou None se faltar alguma linha.
    Só letras maiúsculas isoladas contam ("A alternativa C" dá A e C: reprova, o que é o lado seguro)."""
    cor, con, falta = (_campo(texto, n) for n in ("CORRETAS", "CONTRADI[CÇ][AÃ]O", "FALTA_DADO"))
    sim_nao = lambda v: None if v is None else (
        lambda w: True if w.startswith("SIM") else False if w.startswith(("NAO", "NÃO")) else None)(
        re.sub(r"[^A-ZÃ]", "", v.upper()))
    if falta is not None:
        suf = None if sim_nao(falta) is None else not sim_nao(falta)
    else:  # v1: SUFICIENTE (SIM = ok)
        suf = sim_nao(_campo(texto, "SUFICIENTE"))
    con = sim_nao(con)
    if cor is None or con is None or suf is None:
        return None
    letras = [] if re.sub(r"[^A-Z]", "", cor.upper()).startswith("NENHUMA") else \
        sorted(set(re.findall(r"(?<![A-Za-zÀ-ÿ])([A-E])(?![A-Za-zÀ-ÿ])", cor)))
    return {"corretas": letras, "contradicao": con, "suficiente": suf}


def decidir(texto, modo, item, permutar):
    """(letra ORIGINAL que a conferência sustenta ou None, motivo). None nunca é aprovação."""
    if modo != "critica":
        exib = extrair_letra(texto, modo, letra_nenhuma_exibida(item, permutar))
        return letra_original(exib, item["uid"], permutar, item["alternativas"]), "letra" if exib else "sem_resposta"
    c = extrair_critica(texto)
    if c is None:
        return None, "formato"
    if c["contradicao"]:
        return None, "contradicao"
    if not c["suficiente"]:
        return None, "insuficiente"
    if len(c["corretas"]) != 1:
        return None, "nenhuma_correta" if not c["corretas"] else "varias_corretas"
    return letra_original(c["corretas"][0], item["uid"], permutar, item["alternativas"]), "uma_correta"


def letra_nenhuma_exibida(item, permutar):
    ordem = ordem_alternativas(item["uid"], permutar, item["alternativas"])
    for i, orig in enumerate(ordem):
        if str(item["alternativas"][orig]).strip().lower().startswith("nenhum"):
            return LETRAS[i]
    return None


def letra_original(letra_exibida, uid, permutar, alternativas=None):
    if letra_exibida is None:
        return None
    return ordem_alternativas(uid, permutar, alternativas)[LETRAS.index(letra_exibida)]


def wilson(s, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def apurar_registros(itens, respostas):
    """Pura. itens: rotulados; respostas: {uid: letra_original|None}. Retorna o quadro de decisão."""
    por = {"concorda": [], "discorda": [], "sem_resposta": []}
    for it in itens:
        if it["uid"] not in respostas:
            continue
        r = respostas[it["uid"]]
        k = "sem_resposta" if r is None else ("concorda" if r == it["gabarito_modelo"] else "discorda")
        por[k].append(it)
    validos = lambda xs: sum(bool(x["valida_nucleo"]) for x in xs)
    n = sum(len(v) for v in por.values())
    n_val = sum(validos(v) for v in por.values())
    acc = por["concorda"]
    out = {"n": n, "nucleo_sem_filtro_pct": round(100 * n_val / n, 1) if n else None,
           "mantidas": len(acc), "mantidas_pct": round(100 * len(acc) / n, 1) if n else None,
           "nucleo_entre_mantidas": f"{validos(acc)}/{len(acc)}",
           "nucleo_entre_mantidas_pct": round(100 * validos(acc) / len(acc), 1) if acc else None,
           "ic95_mantidas": wilson(validos(acc), len(acc)),
           "nucleo_entre_descartadas": f"{validos(por['discorda'] + por['sem_resposta'])}/{len(por['discorda']) + len(por['sem_resposta'])}",
           "validas_perdidas_pct": round(100 * (n_val - validos(acc)) / n_val, 1) if n_val else None,
           "invalidas_descartadas_pct": round(100 * (n - n_val - (len(acc) - validos(acc))) / (n - n_val), 1)
           if n - n_val else None}
    ok = [x for x in itens if x["uid"] in respostas and x.get("status_verificador") == "ok"]
    out["verificador_ok"] = f"{validos(ok)}/{len(ok)}"
    # combinação: verificador ok OU concordância
    comb = [x for x in itens if x["uid"] in respostas and (x in acc or x in ok)]
    inter = [x for x in acc if x.get("status_verificador") == "ok"]
    out["concorda_e_ok"] = f"{validos(inter)}/{len(inter)}"
    out["concorda_ou_ok"] = f"{validos(comb)}/{len(comb)}"
    return out


def combinar(itens, *respostas, so_ok=False):
    """Mantidas por TODAS as conferências (e pelo verificador, se so_ok): "válidas/mantidas"."""
    kept = [x for x in itens if all(r.get(x["uid"]) == x["gabarito_modelo"] for r in respostas)
            and (not so_ok or x.get("status_verificador") == "ok")]
    return f"{sum(bool(x['valida_nucleo']) for x in kept)}/{len(kept)}"


def ler_respostas(destino, itens, modo, permutar):
    """{uid: letra_original|None}, {uid: motivo}, relendo o texto gravado com o leitor atual."""
    por_uid = {it["uid"]: it for it in itens}
    resp, motivos = {}, {}
    for l in destino.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            it = por_uid.get(r["uid"])
            if it is not None:
                resp[r["uid"]], motivos[r["uid"]] = decidir(r["texto"], modo, it, permutar)
    return resp, motivos


def cmd_resolver(a):
    import test_model as tm
    itens = obter_itens(a)
    if a.limite:
        itens = itens[:a.limite]
    destino = SAIDA / f"respostas_{sufixo(a)}.jsonl"
    destino.parent.mkdir(parents=True, exist_ok=True)
    feitos = {json.loads(l)["uid"] for l in destino.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if destino.exists() else set()
    tm.LORA_PATH = Path(a.lora) if a.lora else None
    cli = tm.find_llama_cli(None)
    url = tm._Servidor.garantir(cli, a.gguf, tm.DEFAULT_THREADS)
    pend = [it for it in itens if it["uid"] not in feitos]
    print(f"{len(itens)} itens, {len(feitos)} já feitos, {len(pend)} a resolver ({a.modo}, permutar={a.permutar})")
    with open(destino, "a", encoding="utf-8") as f:
        for k, it in enumerate(pend, 1):
            corpo = {"messages": [{"role": "system", "content": SISTEMA_CRITICA if a.modo == "critica" else SISTEMA},
                                  {"role": "user", "content": montar_prompt(it, a.modo, a.permutar)}],
                     "temperature": 0, "max_tokens": MAX_TOKENS[a.modo], "seed": 7, "cache_prompt": True}
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(corpo).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    texto = json.loads(r.read().decode("utf-8", errors="replace"))["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001  falha de servidor = sem resposta, nunca aprovação
                print(f"  erro em {it['uid']}: {e!r}", file=sys.stderr)
                texto = ""
            letra, motivo = decidir(texto, a.modo, it, a.permutar)
            # a crítica guarda mais texto: as três linhas finais e o julgamento das alternativas, para diagnóstico
            f.write(json.dumps({"uid": it["uid"], "letra_original": letra, "motivo": motivo,
                                "texto": texto[-(2000 if a.modo == "critica" else 300):]}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(pend)}")
    print("gravado em", destino)


def cmd_apurar(a):
    destino = SAIDA / f"respostas_{sufixo(a)}.jsonl"
    itens = obter_itens(a)
    resp, motivos = ler_respostas(destino, itens, a.modo, a.permutar)
    res = apurar_registros(itens, resp)
    if a.modo == "critica":
        from collections import Counter
        res["motivos"] = dict(Counter(motivos.values()))
        cot = SAIDA / f"respostas_{sufixo(argparse.Namespace(modo='cot', permutar=a.permutar, rotulo=a.rotulo))}.jsonl"
        if cot.exists():
            resp_cot, _ = ler_respostas(cot, itens, "cot", a.permutar)
            feitos = [x for x in itens if x["uid"] in resp and x["uid"] in resp_cot]
            res["cot_e_critica"] = combinar(feitos, resp, resp_cot)
            res["cot_e_critica_e_ok"] = combinar(feitos, resp, resp_cot, so_ok=True)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (SAIDA / f"apuracao_{sufixo(a)}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nome in ("resolver", "apurar"):
        s = sub.add_parser(nome)
        s.add_argument("--modo", choices=list(FORMATO), default="direto")
        s.add_argument("--permutar", action="store_true")
        s.add_argument("--folha", help="folha preenchida (CSV) em vez de rotulados.jsonl; exige --chave e --rotulo")
        s.add_argument("--chave")
        s.add_argument("--rotulo", default="", help="sufixo dos arquivos de saída (ex.: v8rftb)")
    r = sub.choices["resolver"]
    r.add_argument("--gguf", required=True)
    r.add_argument("--lora", default=None)
    r.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    if a.folha and not (a.chave and a.rotulo):
        ap.error("--folha exige --chave e --rotulo")
    cmd_resolver(a) if a.cmd == "resolver" else cmd_apurar(a)


if __name__ == "__main__":
    main()
