"""Tarefa F, item 4: experimento barato com o juiz Jev (TypeSafe System One, via OpenRouter).

Pergunta: o Jev concorda com o professor em `valida_nucleo` (kappa >= 0,6)? E quais perguntas
isoladas servem como pré-filtro barato?

Regras de segurança:
  * a chave (OPEN_ROUTE_API_KEY) é lida do .env dentro do script e NUNCA é impressa, gravada
    ou incluída em mensagem de erro;
  * teto duro de requisições (padrão 600; `Contador` persiste em jev_contador.json e aborta ANTES
    de enviar a requisição que passaria do teto) e teto de gasto estimado (US$ 0,50). O gasto é a
    soma do `usage.cost` devolvido pela API; requisições sem esse campo contam a tarifa
    conservadora de US$ 0,0003;
  * as respostas brutas vão para jev_respostas.jsonl (uma linha por requisição) e o script
    RETOMA dele: itens já respondidos não são pedidos de novo.

O Jev lê de forma literal e não faz contas. Por isso só há perguntas SEMÂNTICAS, uma
afirmação estreita por pergunta, estado em objeto JSON com campos nomeados, instruções
positivas e critérios alinhados com as instruções.

Uso:
  venv/bin/python src/f_jev.py coletar [--max-requisicoes 600]   # gasta (centavos)
  venv/bin/python src/f_jev.py analisar                          # sem rede
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f_comum import RAIZ, SAIDA  # noqa: E402
import f_estatistica as fe  # noqa: E402

URL = "https://openrouter.ai/api/v1/systemone"
MODELO = "typesafe/jev-1.13"
TETO_REQUISICOES = 600
TETO_GASTO_USD = 0.50
TARIFA_CONSERVADORA = 0.0003
RESPOSTAS = SAIDA / "jev_respostas.jsonl"
CONTADOR = SAIDA / "jev_contador.json"

# Perguntas (alta = sim). `alvo` = dimensão do professor que a pergunta deveria prever;
# `inverter` = sim significa DEFEITO (a pontuação de validade é 1 - p).
PERGUNTAS = OrderedDict([
    ("dados_suficientes", dict(
        alvo="dados_suficientes", inverter=False, completo=True, so_enunciado=True,
        instructions="Does `enunciado` state every number and every fact that is needed to solve the problem?",
        true="Every quantity and fact needed to reach an answer is written in `enunciado`.",
        false="At least one quantity or fact needed to reach an answer is missing from `enunciado`.")),
    ("resposta_unica", dict(
        alvo="resposta_unica", inverter=False, completo=True, so_enunciado=True,
        instructions="Is exactly one option in `alternativas` a correct answer to the problem in `enunciado`?",
        true="Exactly one option is a correct answer.",
        false="No option is a correct answer, or two or more options are correct answers.")),
    ("valor_resolucao_nas_alternativas", dict(
        alvo="correcao_matematica", inverter=False, completo=True, so_enunciado=False,
        instructions="Does the final value or final answer that `resolucao` ends with appear as the text of one of the options in `alternativas`?",
        true="The final value of `resolucao` is the same as the text of one option in `alternativas`.",
        false="The final value of `resolucao` is not the text of any option in `alternativas`.")),
    ("gabarito_coerente_com_resolucao", dict(
        alvo="correcao_matematica", inverter=False, completo=True, so_enunciado=False,
        instructions="Is the text of `resposta_marcada` the same value or answer that `resolucao` ends with?",
        true="`resposta_marcada.texto` is the same value or answer as the end of `resolucao`.",
        false="`resposta_marcada.texto` is a different value or answer than the end of `resolucao`.")),
    ("enunciado_contraditorio", dict(
        alvo="dados_suficientes", inverter=True, completo=True, so_enunciado=True,
        instructions="Does `enunciado` contain two statements that contradict each other?",
        true="Two statements in `enunciado` cannot both be true.",
        false="All statements in `enunciado` can be true at the same time.")),
    ("referencia_a_figura_ausente", dict(
        alvo="dados_suficientes", inverter=True, completo=True, so_enunciado=True,
        instructions="Does `enunciado` refer to a table, graph, figure, image or drawing that is not included in its text?",
        true="`enunciado` mentions a table, graph, figure, image or drawing whose content is not written out.",
        false="`enunciado` does not mention any table, graph, figure, image or drawing, or writes out its content.")),
])


# --------------------------------------------------------------------------
# estado e requisição (puros)
# --------------------------------------------------------------------------
def estado_do_item(item, variante="completo"):
    """Objeto JSON com campos nomeados. `completo` inclui resolução e gabarito marcado;
    `enunciado` só o enunciado e as alternativas (para as perguntas que não precisam do resto)."""
    alts = {l: item["alternativas"].get(l, "") for l in "ABCDE"}
    est = {"enunciado": item["enunciado"], "alternativas": alts}
    if variante == "completo":
        letra = item.get("gabarito_modelo") or ""
        est["resolucao"] = item["resolucao"]
        est["resposta_marcada"] = {"letra": letra, "texto": alts.get(letra, "")}
    return est


def perguntas_da_variante(variante):
    return OrderedDict((k, v) for k, v in PERGUNTAS.items()
                       if variante == "completo" or v["so_enunciado"])


def corpo_requisicao(item, variante="completo"):
    qs = {}
    for k, p in perguntas_da_variante(variante).items():
        qs[k] = {"type": "noul", "instructions": p["instructions"],
                 "criteria": {"true": p["true"], "false": p["false"]}}
    return {"state": estado_do_item(item, variante), "model": MODELO, "questions": qs}


def pontuacao_de_validade(probs, nome):
    """p(sim) -> pontuação em que ALTO = sem defeito, conforme a orientação fixada a priori."""
    p = probs[nome]
    return 1 - p if PERGUNTAS[nome]["inverter"] else p


def combinar_min(probs, nomes):
    """Combinação simples: menor pontuação de validade entre as perguntas escolhidas."""
    vals = [pontuacao_de_validade(probs, n) for n in nomes if n in probs]
    return min(vals) if vals else None


# --------------------------------------------------------------------------
# contador com teto
# --------------------------------------------------------------------------
class TetoExcedido(RuntimeError):
    pass


class Contador:
    """Conta TENTATIVAS HTTP (inclusive as que falham) e o gasto; aborta antes de passar do teto."""

    def __init__(self, caminho=CONTADOR, teto=TETO_REQUISICOES, teto_usd=TETO_GASTO_USD):
        self.caminho, self.teto, self.teto_usd = Path(caminho), teto, teto_usd
        self.lock = threading.Lock()
        estado = {"requisicoes": 0, "gasto_usd": 0.0}
        if self.caminho.exists():
            estado.update(json.loads(self.caminho.read_text(encoding="utf-8")))
        self.requisicoes, self.gasto = int(estado["requisicoes"]), float(estado["gasto_usd"])

    def _gravar(self):
        self.caminho.write_text(json.dumps({"requisicoes": self.requisicoes,
                                            "gasto_usd": round(self.gasto, 6),
                                            "teto_requisicoes": self.teto,
                                            "teto_gasto_usd": self.teto_usd}, indent=2), encoding="utf-8")

    def reservar(self):
        """Chamar ANTES de enviar. Reserva a tarifa conservadora; `liquidar` acerta depois."""
        with self.lock:
            if self.requisicoes >= self.teto:
                raise TetoExcedido(f"teto de {self.teto} requisições atingido")
            if self.gasto + TARIFA_CONSERVADORA > self.teto_usd:
                raise TetoExcedido(f"teto de US$ {self.teto_usd} atingido")
            self.requisicoes += 1
            self.gasto += TARIFA_CONSERVADORA
            self._gravar()

    def liquidar(self, custo_real):
        """Troca a reserva pelo custo informado pela API (se houver)."""
        if custo_real is None:
            return
        with self.lock:
            self.gasto += float(custo_real) - TARIFA_CONSERVADORA
            self._gravar()


def ler_chave():
    """OPEN_ROUTE_API_KEY do .env (ou do ambiente). Nunca é impressa."""
    k = os.environ.get("OPEN_ROUTE_API_KEY")
    if k:
        return k
    env = RAIZ / ".env"
    if env.exists():
        for linha in env.read_text(encoding="utf-8").splitlines():
            if linha.startswith("OPEN_ROUTE_API_KEY="):
                return linha.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("OPEN_ROUTE_API_KEY ausente do .env")


def chamar(corpo, chave, contador, tentativas=3, timeout=60):
    """POST com até `tentativas` envios; CADA envio passa pelo contador. Devolve a resposta (dict)
    ou {"erro": ...} sem vazar a chave."""
    ultimo = None
    for t in range(tentativas):
        contador.reservar()
        req = urllib.request.Request(URL, data=json.dumps(corpo).encode(), method="POST", headers={
            "Authorization": "Bearer " + chave, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                dados = json.loads(r.read().decode())
            contador.liquidar((dados.get("usage") or {}).get("cost"))
            return dados
        except urllib.error.HTTPError as e:
            ultimo = f"HTTP {e.code}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as e:  # noqa: BLE001
            ultimo = type(e).__name__
        time.sleep(2 ** t)
    return {"erro": ultimo}


# --------------------------------------------------------------------------
# coleta
# --------------------------------------------------------------------------
def carregar_unicos():
    itens = [json.loads(l) for l in (SAIDA / "rotulados.jsonl").read_text(encoding="utf-8").splitlines()]
    return [i for i in itens if i["dup_de"] is None]


def ler_respostas():
    if not RESPOSTAS.exists():
        return []
    return [json.loads(l) for l in RESPOSTAS.read_text(encoding="utf-8").splitlines() if l.strip()]


def plano(unicos, respondidos):
    """Ordem de coleta: variante `completo` para todos os itens; depois `enunciado` para os gerados
    por modelo (4B primeiro). `respondidos` = {(sig, variante)}."""
    def prioridade(i):
        return (0 if i["familia"].startswith("qwen_4b") else 1 if i["familia"] != "base_corpus" else 2, i["uid"])
    p = [(i, "completo") for i in sorted(unicos, key=lambda x: x["uid"])]
    gerados = sorted((i for i in unicos if i["familia"] != "base_corpus"), key=prioridade)
    p += [(i, "enunciado") for i in gerados]
    return [(i, v) for i, v in p if (i["sig"], v) not in respondidos]


def coletar(max_req):
    SAIDA.mkdir(parents=True, exist_ok=True)
    contador = Contador(teto=max_req)
    chave = ler_chave()
    unicos = carregar_unicos()
    ja = {(r["sig"], r["variante"]) for r in ler_respostas() if "answers" in r}
    fila = plano(unicos, ja)
    print(f"{len(unicos)} itens únicos; já respondidos {len(ja)}; a pedir {len(fila)}; "
          f"contador {contador.requisicoes}/{contador.teto}, gasto US$ {contador.gasto:.4f}")
    lock = threading.Lock()

    def um(par):
        item, variante = par
        try:
            dados = chamar(corpo_requisicao(item, variante), chave, contador)
        except TetoExcedido as e:
            return "teto", str(e)
        reg = {"uid": item["uid"], "sig": item["sig"], "variante": variante, "ts": int(time.time())}
        if "answers" in dados:
            reg["answers"] = {k: v.get("noul") for k, v in dados["answers"].items()}
            reg["usage"] = dados.get("usage")
        else:
            reg["erro"] = dados.get("erro", "sem answers")
        with lock, open(RESPOSTAS, "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        return ("ok" if "answers" in dados else "erro"), None

    parou = None
    with ThreadPoolExecutor(max_workers=4) as ex:
        for status, msg in ex.map(um, fila):
            if status == "teto":
                parou = msg
    print("coleta encerrada." + (f" ABORTADA PELO TETO: {parou}" if parou else ""))
    print(f"requisições {contador.requisicoes}/{contador.teto}; gasto estimado US$ {contador.gasto:.4f}")
    return 0


# --------------------------------------------------------------------------
# análise
# --------------------------------------------------------------------------
def subconjuntos(unicos):
    return OrderedDict([
        ("todos", lambda i: True),
        ("gerados_por_modelo", lambda i: i["familia"] != "base_corpus"),
        ("qwen_gerados (sem gemma)", lambda i: i["familia"].startswith("qwen")),
        ("4B (ft+zs)", lambda i: i["familia"].startswith("qwen_4b")),
        ("4B ft", lambda i: i["familia"] == "qwen_4b_ft"),
    ])


def linhas_para_analise(unicos, respostas, variante):
    """[(item, {pergunta: p})] só com itens que têm todas as perguntas da variante."""
    por = {}
    for r in respostas:
        if r.get("variante") == variante and "answers" in r:
            por[r["sig"]] = r["answers"]
    nomes = list(perguntas_da_variante(variante))
    out = []
    for i in unicos:
        a = por.get(i["sig"])
        if a and all(isinstance(a.get(n), (int, float)) for n in nomes):
            out.append((i, {n: float(a[n]) for n in nomes}))
    return out


def _alvo_binario(item, alvo):
    return bool(item[alvo])


def avaliar_score(scores, y, reps=1000, seed=0):
    """AUC (IC bootstrap), melhor limiar in-sample por kappa e por acurácia, e kappa/acurácia OOB."""
    a = fe.auc(scores, y)
    if a is None:
        return {"n": len(y), "auc": None}
    lo, hi = fe.bootstrap_auc(scores, y, reps=reps, seed=seed)
    t_k, k_in = fe.melhor_limiar(scores, y, "kappa")
    t_a, acc_in = fe.melhor_limiar(scores, y, "acuracia")
    oob_k = fe.bootstrap_oob(scores, y, "kappa", reps=reps, seed=seed)
    oob_a = fe.bootstrap_oob(scores, y, "acuracia", reps=reps, seed=seed)
    return {"n": len(y), "n_validos": int(sum(y)), "pct_validos": round(100 * sum(y) / len(y), 1),
            "auc": round(a, 3), "auc_ic95": [round(lo, 3), round(hi, 3)],
            "melhor_limiar_kappa": round(t_k, 3), "kappa_in_sample": round(k_in, 3) if k_in is not None else None,
            "melhor_limiar_acuracia": round(t_a, 3), "acuracia_in_sample": round(acc_in, 3),
            "acuracia_base_maioria": round(max(sum(y), len(y) - sum(y)) / len(y), 3),
            "kappa_oob": _arred(oob_k), "acuracia_oob": _arred(oob_a)}


def _arred(d):
    return {"media": None if d["media"] is None else round(d["media"], 3),
            "ic95": [None if x is None else round(x, 3) for x in d["ic95"]],
            "limiar_mediano": None if d["limiar_mediano"] is None else round(d["limiar_mediano"], 3)}


def tabela_prefiltro(scores, y, limiares=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95)):
    """Uso como pré-filtro: reprova (flag) se score < t. Precisão = fração dos reprovados que são
    de fato inválidos; recall = fração dos inválidos pegos; custo = fração dos válidos reprovados."""
    inval = [not v for v in y]
    n_inv, n_val = sum(inval), sum(1 for v in y if v)
    out = []
    for t in limiares:
        flag = [s < t for s in scores]
        tp = sum(1 for f, iv in zip(flag, inval) if f and iv)
        fp = sum(1 for f, iv in zip(flag, inval) if f and not iv)
        out.append({"reprova_se_score_menor_que": t, "n_reprovados": tp + fp,
                    "precisao_invalido": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall_invalido": round(tp / n_inv, 3) if n_inv else None,
                    "validos_perdidos_pct": round(100 * fp / n_val, 1) if n_val else None})
    return out


def analisar(reps=1000, seed=0):
    unicos = carregar_unicos()
    respostas = ler_respostas()
    contador = Contador() if CONTADOR.exists() else None
    resultado = OrderedDict()
    resultado["meta"] = {
        "modelo": MODELO, "n_unicos_rotulados": len(unicos),
        "n_respostas_ok": sum(1 for r in respostas if "answers" in r),
        "n_respostas_erro": sum(1 for r in respostas if "answers" not in r),
        "requisicoes_http": contador.requisicoes if contador else None,
        "gasto_estimado_usd": round(contador.gasto, 4) if contador else None,
        "bootstrap_reps": reps, "seed": seed,
        "criterio_do_juiz": "Cohen kappa >= 0,6 com o professor em valida_nucleo",
        "convencao": "positivo = valida_nucleo; pontuação ALTA = sem defeito"}
    combos = OrderedDict([
        ("min(4 semanticas de conteudo)", ["dados_suficientes", "resposta_unica",
                                           "valor_resolucao_nas_alternativas", "gabarito_coerente_com_resolucao"]),
        ("min(6 perguntas)", list(PERGUNTAS)),
        ("min(dados_suficientes, resposta_unica)", ["dados_suficientes", "resposta_unica"]),
        ("min(dados_suficientes, resposta_unica, sem contradicao, sem figura)",
         ["dados_suficientes", "resposta_unica", "enunciado_contraditorio", "referencia_a_figura_ausente"]),
    ])
    for variante in ("completo", "enunciado"):
        linhas = linhas_para_analise(unicos, respostas, variante)
        if not linhas:
            continue
        bloco = OrderedDict()
        vistos = set()
        for nome_sub, filtro in subconjuntos(unicos).items():
            sub = [(i, p) for i, p in linhas if filtro(i)]
            conj = frozenset(i["uid"] for i, _ in sub)
            if len(sub) < 10 or conj in vistos:  # subconjunto idêntico a um anterior: não repete
                continue
            vistos.add(conj)
            y_nuc = [bool(i["valida_nucleo"]) for i, _ in sub]
            res = OrderedDict(n=len(sub), n_validos_nucleo=int(sum(y_nuc)))
            por_pergunta = OrderedDict()
            for nome in perguntas_da_variante(variante):
                sc = [pontuacao_de_validade(p, nome) for _, p in sub]
                alvo = PERGUNTAS[nome]["alvo"]
                por_pergunta[nome] = {
                    "vs_valida_nucleo": avaliar_score(sc, y_nuc, reps, seed),
                    f"vs_{alvo}": avaliar_score(sc, [_alvo_binario(i, alvo) for i, _ in sub], reps, seed),
                    "prefiltro_vs_valida_nucleo": tabela_prefiltro(sc, y_nuc),
                    "prefiltro_vs_alvo": tabela_prefiltro(sc, [_alvo_binario(i, alvo) for i, _ in sub]),
                    "alvo": alvo}
            res["por_pergunta"] = por_pergunta
            res["combinacoes"] = OrderedDict()
            for cn, nomes in combos.items():
                if not all(n in perguntas_da_variante(variante) for n in nomes):
                    continue
                sc = [combinar_min(p, nomes) for _, p in sub]
                res["combinacoes"][cn] = {"vs_valida_nucleo": avaliar_score(sc, y_nuc, reps, seed),
                                          "prefiltro_vs_valida_nucleo": tabela_prefiltro(sc, y_nuc)}
            bloco[nome_sub] = res
        resultado[f"variante_{variante}"] = bloco
    return resultado


def melhor_kappa_oob(resultado):
    """(kappa_oob_media, rotulo) do melhor candidato (pergunta ou combinação) em gerados_por_modelo/completo."""
    melhor = (None, None)
    bloco = resultado.get("variante_completo", {}).get("gerados_por_modelo")
    if not bloco:
        return melhor
    cands = {**{f"pergunta:{k}": v["vs_valida_nucleo"] for k, v in bloco["por_pergunta"].items()},
             **{f"combinacao:{k}": v["vs_valida_nucleo"] for k, v in bloco["combinacoes"].items()}}
    for rot, r in cands.items():
        k = (r.get("kappa_oob") or {}).get("media")
        if k is not None and (melhor[0] is None or k > melhor[0]):
            melhor = (k, rot)
    return melhor


def veredicto_md(res):
    """Texto objetivo sobre o critério kappa >= 0,6, sem escolher o 'melhor' depois de ver os dados:
    lista todos os candidatos e marca os que passam (média OOB) e os que passam com IC inferior >= 0,6."""
    L = ["## Veredicto (critério: kappa >= 0,6 com o professor em `valida_nucleo`)", ""]
    for sub in ("gerados_por_modelo", "qwen_gerados (sem gemma)", "4B (ft+zs)", "4B ft"):
        r = res.get("variante_completo", {}).get(sub)
        if not r:
            continue
        cand = {**{k: v["vs_valida_nucleo"] for k, v in r["por_pergunta"].items()},
                **{k: v["vs_valida_nucleo"] for k, v in r["combinacoes"].items()}}
        passa = [k for k, v in cand.items() if (v.get("kappa_oob") or {}).get("media") is not None
                 and v["kappa_oob"]["media"] >= 0.6]
        firme = [k for k in passa if cand[k]["kappa_oob"]["ic95"][0] is not None
                 and cand[k]["kappa_oob"]["ic95"][0] >= 0.6]
        L.append(f"- **{sub}** (n={r['n']}): κ OOB médio >= 0,6 em {len(passa)} de {len(cand)} candidatos "
                 f"({'; '.join(passa) or 'nenhum'}); com o limite inferior do IC95 também >= 0,6: "
                 f"{'; '.join(firme) or 'nenhum'}.")
    L += ["", "Ressalvas: (1) escolher o melhor de 10 candidatos infla a estimativa; as perguntas e as combinações foram "
          "fixadas ANTES de olhar os dados, e todas estão listadas acima. (2) Os negativos vêm principalmente de 1.7B e "
          "Gemma; o subconjunto 4B tem só 53 itens (30 do 4B fine-tunado) e ICs largos. (3) O limiar é aprendido nos "
          "próprios rótulos do professor (por isso OOB); para uso real, congelar o limiar e validar numa amostra nova "
          "(a folha cega de 222 itens do 4B serve para isso).", ""]
    return "\n".join(L)


def relatorio_md(res):
    m = res["meta"]
    L = ["# Calibração do juiz Jev (jev-1.13) contra o professor", "",
         f"- itens rotulados únicos: {m['n_unicos_rotulados']}; respostas ok {m['n_respostas_ok']}, "
         f"com erro {m['n_respostas_erro']}",
         f"- requisições HTTP: {m['requisicoes_http']}; gasto estimado: US$ {m['gasto_estimado_usd']}",
         f"- critério: {m['criterio_do_juiz']}. Convenção: {m['convencao']}.",
         f"- bootstrap: {m['bootstrap_reps']} reamostragens, seed {m['seed']}. `kappa_oob`/`acuracia_oob` reajustam o "
         "limiar em cada reamostra e medem nos itens fora dela (estimativa honesta); `*_in_sample` ajustam e medem "
         "nos mesmos dados (otimista).", ""]
    L += [veredicto_md(res)]
    for var in ("variante_completo", "variante_enunciado"):
        if var not in res:
            continue
        L.append(f"## {var.replace('_', ' ')}")
        for sub, r in res[var].items():
            L += ["", f"### subconjunto: {sub} (n={r['n']}, válidos pelo núcleo: {r['n_validos_nucleo']})", "",
                  "| pergunta / combinação | AUC [IC95] | κ in-sample | κ OOB [IC95] | acurácia OOB | base (maioria) | alvo |",
                  "|---|---|---|---|---|---|---|"]
            linhas = [(k, v["vs_valida_nucleo"], v["alvo"]) for k, v in r["por_pergunta"].items()]
            linhas += [(f"**{k}**", v["vs_valida_nucleo"], "-") for k, v in r["combinacoes"].items()]
            for k, v, alvo in linhas:
                if v.get("auc") is None:
                    L.append(f"| {k} | n/d | | | | | {alvo} |")
                    continue
                L.append(f"| {k} | {v['auc']} [{v['auc_ic95'][0]}; {v['auc_ic95'][1]}] | {v['kappa_in_sample']} | "
                         f"{v['kappa_oob']['media']} [{v['kappa_oob']['ic95'][0]}; {v['kappa_oob']['ic95'][1]}] | "
                         f"{v['acuracia_oob']['media']} | {v['acuracia_base_maioria']} | {alvo} |")
            L += ["", "Pré-filtro (reprova o item se a pontuação de validade < t; alvo `valida_nucleo`; "
                  "precisão = % dos reprovados realmente inválidos; recall = % dos inválidos pegos; "
                  "perdidos = % dos válidos reprovados por engano):", "",
                  "| candidato | t | reprovados | precisão | recall | válidos perdidos % |", "|---|---|---|---|---|---|"]
            cand_pf = [(k, v["prefiltro_vs_valida_nucleo"]) for k, v in r["por_pergunta"].items()
                       if k in ("resposta_unica", "dados_suficientes", "gabarito_coerente_com_resolucao",
                                "valor_resolucao_nas_alternativas")]
            cand_pf += [(k, v["prefiltro_vs_valida_nucleo"]) for k, v in r["combinacoes"].items()
                        if k.startswith("min(dados_suficientes, resposta_unica)") or k.startswith("min(4 ")]
            for k, tab in cand_pf:
                for t in tab:
                    L.append(f"| {k} | {t['reprova_se_score_menor_que']} | {t['n_reprovados']} | {t['precisao_invalido']} | "
                             f"{t['recall_invalido']} | {t['validos_perdidos_pct']} |")
            L += ["", "Cada pergunta contra a SUA dimensão do professor:", "",
                  "| pergunta | alvo | AUC [IC95] | κ OOB | acurácia OOB | base (maioria) |", "|---|---|---|---|---|---|"]
            for k, v in r["por_pergunta"].items():
                w = v[f"vs_{v['alvo']}"]
                if w.get("auc") is None:
                    continue
                L.append(f"| {k} | {v['alvo']} | {w['auc']} [{w['auc_ic95'][0]}; {w['auc_ic95'][1]}] | "
                         f"{w['kappa_oob']['media']} | {w['acuracia_oob']['media']} | {w['acuracia_base_maioria']} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coletar")
    c.add_argument("--max-requisicoes", type=int, default=TETO_REQUISICOES)
    a = sub.add_parser("analisar")
    a.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd == "coletar":
        return coletar(min(args.max_requisicoes, TETO_REQUISICOES))
    res = analisar(args.reps)
    (SAIDA / "calibracao_jev.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (SAIDA / "calibracao_jev.md").write_text(relatorio_md(res), encoding="utf-8")
    k, rot = melhor_kappa_oob(res)
    print(f"melhor kappa OOB (gerados por modelo, estado completo): {k} ({rot})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
