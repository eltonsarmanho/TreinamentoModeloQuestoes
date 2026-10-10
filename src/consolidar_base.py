"""Consolida a base de treino: remove o que não se sustenta, refaz o que falta e balanceia as letras.

Entrada: data/train_curado_v4.jsonl (v3 com texto padronizado por src/padronizar_enunciado.py).
Saída:   data/train_curado_v5.jsonl + outputs/relatorios/consolidacao_v5_log.jsonl.
Nunca escreve em train_curado.jsonl, train.jsonl, val*.jsonl nem no DB/questoes.db.

Fases (cada uma com cache em outputs/agentes/consolidacao_cache.jsonl e TETO DURO de chamadas):

  figura   (sabia-4)  — para os itens com termo visual (figura, observe, abaixo, mapa...) que NÃO
                        trazem os dados em texto: "dá para resolver só com o texto?". Quem depende de
                        figura que o app não tem é removido.
  resolver (sabia-4-thinking) — para os itens reais do banco cujas justificativas oficiais NÃO marcam
                        a alternativa "Correto" (473 no banco; os que estão na base): lê o enunciado,
                        resolve às cegas em 2 permutações das alternativas, escolhe a alternativa e
                        produz a resolução passo a passo.
  montar   (sem API)  — aplica as regras abaixo, balanceia a letra do gabarito (permutar_lote) e grava.

Regras de decisão da fase resolver (por item):
  * os 2 passes concordam entre si e com o gabarito  -> mantém o gabarito; troca a resolução pela nova
    se ela passar nas validações (>= 40 caracteres, não cita letra de alternativa, sem CAIXA ALTA, o
    verificador aritmético estendido não a reprova nem acusa resultado fora das alternativas);
  * os 2 passes concordam entre si e DISCORDAM do gabarito -> só adota a letra nova se o verificador
    aritmético CONFIRMAR (ok=True) a nova resolução contra a nova letra; senão o item é REMOVIDO;
  * os passes discordam, ou nenhuma/mais de uma alternativa serve ("X"), ou falta resposta -> REMOVIDO.
Também são removidos: os "inconclusivos" de PROPOSTAS_CORRECAO_GABARITO.jsonl e os dois itens cuja
troca foi proposta (ambos dependem de figura ou admitem duas leituras).

Exemplo com mais de uma questão: se QUALQUER questão do exemplo é removida, o exemplo inteiro sai
(o prompt pede "N questões"; deixar N-1 o tornaria falso).

Uso:
    python src/consolidar_base.py figura   --real --max-chamadas 150
    python src/consolidar_base.py resolver --real --max-chamadas 600
    python src/consolidar_base.py montar
"""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import schema_utils as su  # noqa: E402
import diversidade as dv  # noqa: E402
from padronizar_enunciado import eh_caixa_alta  # noqa: E402
from reverificar_gabaritos import permutacoes, montar_pergunta  # noqa: E402

ENTRADA = ROOT / "data" / "train_curado_v4.jsonl"
SAIDA = ROOT / "data" / "train_curado_v5.jsonl"
LOG = ROOT / "outputs" / "relatorios" / "consolidacao_v5_log.jsonl"
PROPOSTAS = ROOT / "outputs" / "relatorios" / "PROPOSTAS_CORRECAO_GABARITO.jsonl"
CACHE = ROOT / "outputs" / "agentes" / "consolidacao_cache.jsonl"
FIGURA_JSON = ROOT / "outputs" / "relatorios" / "consolidacao_figura.json"
RESOLVER_JSON = ROOT / "outputs" / "relatorios" / "consolidacao_resolver.json"
SEED_LETRAS = 20261010

TERMO_VISUAL = re.compile(
    r"\b(figura|imagem|desenho|ilustra\w*|esquema|mapa|malha|croqui|planta|gr[áa]fico|tabela|quadro|"
    r"reta num[ée]rica|abaixo|acima|ao lado|a seguir|veja|observe)\b", re.I)
REF_LETRA = re.compile(r"\b(alternativa|letra|op[cç][aã]o|item)\s+\(?[A-E]\)?(?![\wà-ÿ])|^\s*\(?[A-E]\)\s", re.I | re.M)

SISTEMA_FIGURA = (
    "Você avalia questões de matemática do ensino básico brasileiro que serão apresentadas num aplicativo "
    "SÓ COM TEXTO (sem figura, imagem, gráfico, mapa ou tabela desenhada). Diga se a questão é resolvível "
    "apenas com o texto do enunciado e das alternativas, isto é, se todos os dados necessários estão escritos. "
    "Uma figura apenas DESCRITA em palavras com todos os dados necessários conta como resolvível. Se a questão "
    "manda olhar uma figura/relógio/tabela/gráfico/planificação que NÃO está descrito, ou se as alternativas "
    'são desenhos, NÃO é resolvível. Responda APENAS com JSON: {"resolvivel_so_com_texto": true|false, "falta": "<o que falta, ou vazio>"}.')

SISTEMA_RESOLVER = (
    "Você é um professor de matemática do ensino básico brasileiro. Resolva a questão do zero, com rigor, e "
    "escolha a alternativa correta. Escreva a resolução passo a passo em português, com as contas explícitas "
    "no formato 'a + b = c' (use os sinais + - x ÷), SEM citar letras de alternativas (cite valores), terminando "
    "com a conclusão. Se nenhuma alternativa estiver correta, ou mais de uma estiver, responda \"X\" e explique "
    'em "resolucao". Responda APENAS com JSON: {"resposta": "A|B|C|D|E|X", "resolucao": "<passo a passo>"}.')


class Chamador:
    def __init__(self, modelo, cliente, max_chamadas, timeout_msg="max_chamadas"):
        self.modelo, self.cliente, self.max_chamadas, self.chamadas = modelo, cliente, max_chamadas, 0
        self.esgotado = self.falhou = False  # sem resposta por teto/API NÃO pode virar "item removido"
        self.cache = {}
        if CACHE.exists():
            for l in CACHE.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    self.cache[r["chave"]] = r["texto"]

    def __call__(self, sistema, payload, max_tokens):
        chave = hashlib.sha256((self.modelo + "\x1e" + sistema + "\x1e" + payload).encode("utf-8")).hexdigest()
        if chave in self.cache:
            return self.cache[chave]
        if self.chamadas >= self.max_chamadas:
            self.esgotado = True
            return None
        self.chamadas += 1
        for t in range(3):
            try:
                r = self.cliente.chat_completion([{"role": "system", "content": sistema},
                                                  {"role": "user", "content": payload}],
                                                 max_tokens=max_tokens, temperature=0.0)
                txt = r.choices[0].message.content
                self.cache[chave] = txt
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"chave": chave, "modelo": self.modelo, "texto": txt}, ensure_ascii=False) + "\n")
                return txt
            except Exception:  # noqa: BLE001 — rede/cota: tenta de novo; a chave nunca é impressa
                time.sleep(3 * (t + 1))
        self.falhou = True
        return None

    def checar(self):
        if self.esgotado or self.falhou:
            raise SystemExit("ABORTADO: teto de chamadas esgotado ou falha de API; nada foi decidido por falta de "
                             "resposta. Aumente --max-chamadas (o cache evita refazer o que já foi pago) e rode de novo.")


def _json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- leitura


def carregar_base(caminho=ENTRADA):
    """Lista de (linha, indice_questao, meta, questao)."""
    out = []
    for i, l in enumerate(open(caminho, encoding="utf-8")):
        if not l.strip():
            continue
        r = json.loads(l)
        for j, q in enumerate(json.loads(r["messages"][2]["content"])["questoes"]):
            out.append((i, j, r["meta"], q))
    return out


def sem_marca_no_banco(db=ROOT / "DB" / "questoes.db"):
    """codigo_item dos itens do banco SEM exatamente uma justificativa 'Correto' (473 no banco)."""
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    out = set()
    for x in con.execute("select * from itens where gabarito in ('A','B','C','D')"):
        j = {L: (x[f"justificativa_alternativa_{L.lower()}"] or "").strip() for L in "ABCD"}
        if len([L for L, v in j.items() if v.lower().startswith(("correto", "correta"))]) != 1:
            out.add(x["codigo_item"])
    return out


def candidatos_figura(base):
    """Itens com termo visual e SEM dados em texto (tem_dados_textuais), mais os do detector do projeto
    e os da lista de propostas de troca (dependem de figura/ambíguos)."""
    prop = set()
    if PROPOSTAS.exists():
        prop = {json.loads(l)["codigo_item"] for l in open(PROPOSTAS, encoding="utf-8")
                if json.loads(l)["classe"] == "proposta_trocar"}
    out = []
    for i, j, m, q in base:
        t = q["enunciado"]
        if m["codigo_item"] in prop or su.depende_de_visual_ausente(t) or (TERMO_VISUAL.search(t) and not dv.tem_dados_textuais(q)):
            out.append((i, j, m, q))
    return out


# --------------------------------------------------------------------------- fase figura


def fase_figura(base, chamador):
    res = {}
    for i, j, m, q in candidatos_figura(base):
        payload = q["enunciado"] + "\n\n" + "\n".join(f"{L}) {v}" for L, v in q["alternativas"].items())
        r = _json(chamador(SISTEMA_FIGURA, payload, 400))
        chamador.checar()
        res[m["codigo_item"]] = {"linha": i, "questao": j, "resolvivel": None if r is None else bool(r.get("resolvivel_so_com_texto")),
                                 "falta": (r or {}).get("falta", "")}
    return res


# --------------------------------------------------------------------------- fase resolver


def _resolucao_valida(q, nova_resp, resolucao):
    """Validações determinísticas da resolução nova contra a letra proposta. Retorna (ok, motivo, verificador)."""
    if not isinstance(resolucao, str) or len(resolucao.strip()) < 40:
        return False, "resolucao_curta", None
    if REF_LETRA.search(resolucao):
        return False, "cita_letra_de_alternativa", None
    if eh_caixa_alta(resolucao):
        return False, "caixa_alta", None
    t = dict(q, resposta_correta=nova_resp, resolucao_passo_a_passo=resolucao.strip())
    ok, _sug, motivo = su.check_consistency_detalhado(t)
    if ok is False or su.resposta_fora_das_alternativas(t):
        return False, f"verificador_reprova({motivo})", False
    return True, "ok", ok  # ok True = confirmado; None = não verificável


def resolver_item(q, chamador, seed):
    """Dict com decisão para um item. Não altera q."""
    passes = []
    for perm in permutacoes(q, seed):
        r = _json(chamador(SISTEMA_RESOLVER, montar_pergunta(q, perm), 8000))
        resp = None
        if r is not None:
            letra = str(r.get("resposta", "")).strip().upper()
            if letra == "X":
                resp = "X"
            elif letra in "ABCDE" and len(letra) == 1 and "ABCDE".index(letra) < len(perm):
                resp = perm["ABCDE".index(letra)]
        passes.append({"resposta": resp, "resolucao": (r or {}).get("resolucao")})
    a, b = passes[0]["resposta"], passes[1]["resposta"]
    gab = q["resposta_correta"]
    if a is None or b is None or a != b or a == "X":
        return {"decisao": "remover", "motivo": "passes_discordam_ou_sem_resposta", "passes": [p["resposta"] for p in passes]}
    # resolução: tenta a do passe 1, depois a do passe 2
    for p in passes:
        ok, motivo, verif = _resolucao_valida(q, a, p["resolucao"])
        if ok:
            if a == gab:
                return {"decisao": "manter", "gabarito": gab, "resolucao": p["resolucao"].strip(), "verificador": verif,
                        "passes": [x["resposta"] for x in passes]}
            if verif is True:
                return {"decisao": "trocar_gabarito", "gabarito": a, "resolucao": p["resolucao"].strip(), "verificador": True,
                        "gabarito_antigo": gab, "passes": [x["resposta"] for x in passes]}
            return {"decisao": "remover", "motivo": "discorda_do_gabarito_sem_confirmacao_aritmetica", "passes": [x["resposta"] for x in passes]}
    if a == gab:  # gabarito confirmado às cegas, mas nenhuma resolução nova passou nas validações
        return {"decisao": "manter_resolucao_antiga", "gabarito": gab, "motivo": motivo, "passes": [x["resposta"] for x in passes]}
    return {"decisao": "remover", "motivo": f"discorda_do_gabarito_e_resolucao_invalida({motivo})", "passes": [x["resposta"] for x in passes]}


def fase_resolver(base, chamador):
    alvo_codigos = sem_marca_no_banco()
    res = {}
    for k, (i, j, m, q) in enumerate(x for x in base if x[2]["codigo_item"] in alvo_codigos):
        d = resolver_item(q, chamador, seed=k)
        chamador.checar()
        d.update({"linha": i, "questao": j})
        res[m["codigo_item"]] = d
        print(f"[{k + 1}] {m['codigo_item']} {d['decisao']} {d.get('motivo', '')} {d['passes']}", flush=True)
    return res


# --------------------------------------------------------------------------- montar


def montar(base_path=ENTRADA, saida=SAIDA, figura=None, resolver=None, propostas=PROPOSTAS, seed=SEED_LETRAS, log=LOG):
    from gerar_lote import permutar_lote
    figura = figura if figura is not None else json.loads(FIGURA_JSON.read_text(encoding="utf-8"))
    resolver = resolver if resolver is not None else json.loads(RESOLVER_JSON.read_text(encoding="utf-8"))
    remover, motivos = {}, {}

    def marca(cod, motivo):
        remover.setdefault(cod, motivo)

    if propostas and Path(propostas).exists():
        for l in open(propostas, encoding="utf-8"):
            r = json.loads(l)
            if r["classe"] in ("inconclusivo", "proposta_trocar"):
                marca(r["codigo_item"], "inconclusivo_na_resolucao_cega" if r["classe"] == "inconclusivo" else "dependente_de_figura_ou_ambiguo")
    for cod, f in figura.items():
        if f["resolvivel"] is False:
            marca(cod, "depende_de_figura")
        elif f["resolvivel"] is None:
            marca(cod, "figura_nao_avaliada")  # sem veredito => não certificamos
    for cod, d in resolver.items():
        if d["decisao"] == "remover":
            marca(cod, f"resolucao_cega:{d['motivo']}")

    linhas = [json.loads(l) for l in open(base_path, encoding="utf-8") if l.strip()]
    log_regs, saida_linhas = [], []
    for i, r in enumerate(linhas):
        obj = json.loads(r["messages"][2]["content"])
        cods = [r["meta"]["codigo_item"]] * len(obj["questoes"])
        if any(c in remover for c in cods):
            log_regs.append({"codigo_item": r["meta"]["codigo_item"], "acao": "removido", "motivo": remover[r["meta"]["codigo_item"]]})
            continue
        for j, q in enumerate(obj["questoes"]):
            d = resolver.get(r["meta"]["codigo_item"])
            if d and d["decisao"] in ("manter", "trocar_gabarito"):
                antigo = {"gabarito": q["resposta_correta"], "resolucao": q.get("resolucao_passo_a_passo")}
                q["resposta_correta"] = d["gabarito"]
                q["resolucao_passo_a_passo"] = d["resolucao"]
                log_regs.append({"codigo_item": r["meta"]["codigo_item"], "acao": d["decisao"], "antes": antigo,
                                 "depois": {"gabarito": d["gabarito"], "resolucao": d["resolucao"]}, "verificador": d.get("verificador")})
            elif d and d["decisao"] == "manter_resolucao_antiga":
                log_regs.append({"codigo_item": r["meta"]["codigo_item"], "acao": "manteve_resolucao_antiga", "motivo": d.get("motivo")})
        r["messages"][2]["content"] = json.dumps(obj, ensure_ascii=False)
        saida_linhas.append(r)

    # balanceamento de letras: um único lote com a base inteira (guloso, determinístico, com vetos)
    ref = [(k, j) for k, r in enumerate(saida_linhas) for j in range(len(json.loads(r["messages"][2]["content"])["questoes"]))]
    objs = [json.loads(r["messages"][2]["content"]) for r in saida_linhas]
    flat = [objs[k]["questoes"][j] for k, j in ref]
    antes = Counter(q["resposta_correta"] for q in flat)
    novas = permutar_lote(flat, seed)
    trocadas = 0
    for (k, j), nova in zip(ref, novas):
        trocadas += nova["resposta_correta"] != objs[k]["questoes"][j]["resposta_correta"]
        objs[k]["questoes"][j] = nova
    for r, o in zip(saida_linhas, objs):
        r["messages"][2]["content"] = json.dumps(o, ensure_ascii=False)
    depois = Counter(q["resposta_correta"] for o in objs for q in o["questoes"])

    saida = Path(saida)
    if saida.name in ("train_curado.jsonl", "train.jsonl") or saida.name.startswith("val") or saida.resolve() == Path(base_path).resolve():
        raise SystemExit(f"ABORTADO: não escrevo em {saida.name}.")
    saida.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in saida_linhas), encoding="utf-8")
    Path(log).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log_regs), encoding="utf-8")
    return {"exemplos_entrada": len(linhas), "exemplos_saida": len(saida_linhas),
            "removidos": dict(Counter(v.split(":")[0].split("(")[0] for v in remover.values())),
            "gabaritos_trocados": sum(1 for x in log_regs if x.get("acao") == "trocar_gabarito"),
            "resolucoes_novas": sum(1 for x in log_regs if x.get("acao") in ("manter", "trocar_gabarito")),
            "letras_antes": dict(sorted(antes.items())), "letras_depois": dict(sorted(depois.items())),
            "questoes_com_letra_trocada": trocadas}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fase", choices=["figura", "resolver", "montar"])
    ap.add_argument("--base", default=str(ENTRADA))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-chamadas", type=int, default=0)
    args = ap.parse_args()
    base = carregar_base(args.base)
    if args.fase == "montar":
        print(json.dumps(montar(args.base), ensure_ascii=False, indent=2))
        return
    if args.fase == "figura":
        n = len(candidatos_figura(base))
        modelo, destino = "sabia-4", FIGURA_JSON
    else:
        n = sum(1 for x in base if x[2]["codigo_item"] in sem_marca_no_banco())
        modelo, destino = "sabia-4-thinking", RESOLVER_JSON
    print(f"{n} itens na fase {args.fase}; chamadas previstas: {n if args.fase == 'figura' else 2 * n}")
    if not args.real:
        print("Simulação: nenhuma chamada. Use --real --max-chamadas N.")
        return
    if args.max_chamadas <= 0:
        sys.exit("ABORTADO: --real exige --max-chamadas N > 0 (teto duro).")
    import distill_teacher as dt  # carrega o .env; a chave nunca é impressa
    ch = Chamador(modelo, dt.MaritacaClient(modelo, timeout=300), args.max_chamadas)
    res = fase_figura(base, ch) if args.fase == "figura" else fase_resolver(base, ch)
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("chamadas feitas:", ch.chamadas, "| itens:", len(res),
          "| contagem:", dict(Counter((v.get("decisao") or str(v.get("resolvivel"))) for v in res.values())))


if __name__ == "__main__":
    main()
