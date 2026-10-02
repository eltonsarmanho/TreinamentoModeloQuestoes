"""Calibração dos agentes VALIDADOR e REVISOR com chamadas reais (Etapa 3).

Por que um script próprio (e não só `auditar_base.py --calibrar`): o gate de
lá mede só as 20 questões ruins/boas do 9º H17. Um juiz que reprova tudo
passaria nele. Aqui medimos também a FALSA REJEIÇÃO em itens reais do banco
(DB/questoes.db), presumidamente corretos, e registramos cada rodada de ajuste
de prompt com o custo exato.

Conjuntos:
  cal20  — R1/R2 (outputs/testes_locais), rótulos humanos do Log.txt
           (APROVAR R1-Q1, R1-Q7, R2-Q1, R2-Q5, R2-Q9; REPROVAR as outras 15).
  real30 — 30 itens do banco convertidos pelo MESMO build_example do treino,
           sem imagem, sem descrição de imagem, sem remissão a figura no texto
           e com justificativa (a fase 2 do validador lê a resolução). 6 vêm
           do conjunto de geometria plana (onde a regra de hierarquia pode
           sobrecorrigir), 24 estratificados por ano. Sorteio com semente fixa.

Custo (a API cobra por chamada): todo envio passa pelo Orcamento do
agentes_questoes, com teto = min(--max-chamadas, teto global - já gasto).
O "já gasto" fica em outputs/agentes/calibracao_ledger.json e sobrevive a
quedas, então a soma de TODAS as execuções nunca passa de --teto-global (160).
Respostas bem-sucedidas vão para um cache por (papel, modelo, max_tokens,
temperatura, mensagens): rodar de novo com o mesmo prompt custa zero, e numa
rodada de ajuste só o juiz cujo prompt mudou é chamado de novo. Temperatura 0
nos juízes, então reaproveitar a resposta é equivalente a repetir a chamada.

Saída: outputs/calibracao_agentes.json (rodadas, matrizes de confusão, custo,
itens) e outputs/agentes/calibracao_prompts/<versao>.json (texto exato dos
prompts de cada rodada, para auditoria).

Recalibração de 2026-10-01 (decisões D1-D5/T5; rodadas r5 e r6): quatro
conjuntos — cal20, real30, "defeitos" (defeitos conhecidos da auditoria do
piloto, DEFEITOS_CONHECIDOS) e "d5" (data/calibracao_d5.json: 6 distratores
verdadeiros por outro eixo + 6 controles). r5 = prompts d685c9b1e5fd; r6 =
prompts 788dcddaf0e6 (ajuste geral nos dois juízes), com 13 ruins do cal20
herdadas de r5 (--herdar r5). Orçamento da etapa: 200 chamadas (ledger 180 -> 380):
    python src/calibrar_agentes.py --rodada r6 --conjunto defeitos --max-chamadas 18 --teto-global 380
    python src/calibrar_agentes.py --rodada r6 --metricas --herdar r5 --conjunto todos   # 0 chamadas

Passo 2 da recalibração (2026-10-01 tarde; rodada r7, prompts sem a âncora
da dificuldade pedida — P3): conjunto "dif45" = 45 itens REAIS do banco com a
dificuldade no codigo_item (MT9018MH04MT -> letra M = Moderado; F/M/D batendo
com grau_resolucao), 15 F/15 M/15 D, 2º/5º/9º ano. Só o REVISOR é chamado neles
(1 chamada por item): mede a concordância da dificuldade_real com o rótulo do
banco (matriz de confusão, kappa ponderado, taxa de "rebaixaria para Fácil").
O árbitro Gemini (--gemini-dificuldade, orçamento PRÓPRIO) julga a mesma
dificuldade pela mesma rubrica nos itens que a D3 poderia rebaixar (banco M/D)
e numa amostra de F, para medir a regra "2 de 2". A regra D3 sai desses
números (regra_d3). Orçamento da etapa: 160 Maritaca + 40 Gemini (gasto:
121 Maritaca, 38 Gemini = 2 árbitro da D2 + 36 dificuldade). Resultado: D3
SUSPENSA (revisor rebaixaria 83% dos M/D reais; 2 de 2, 41%). Comandos usados:
    python src/calibrar_agentes.py --rodada r7 --conjunto dif --max-chamadas 50 --teto-global 533
    python src/calibrar_agentes.py --rodada r7 --gemini-dificuldade --max-chamadas-gemini 38   # (retomado com
    python src/calibrar_agentes.py --rodada r7 --gemini-dificuldade --max-chamadas-gemini 19 --teto-gemini 38)
    python src/calibrar_agentes.py --rodada r7 --conjunto regressao --max-chamadas 95 --teto-global 533
    python src/calibrar_agentes.py --rodada r7 --metricas --conjunto regressao   # 0 chamadas

Uso:
    python src/calibrar_agentes.py --listar-real            # 0 chamadas
    python src/calibrar_agentes.py --rodada r0 --conjunto cal --max-chamadas 60
    python src/calibrar_agentes.py --rodada r0 --conjunto real --max-chamadas 100
    python src/calibrar_agentes.py --rodada r0 --metricas    # recalcula, 0 chamadas
    python src/calibrar_agentes.py --dry-run --rodada teste --conjunto ambos
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import agentes_questoes as aq
import arbitro_gemini as ag
import auditar_base as ab
from extract_data import build_example, load_rows
from schema_utils import IMAGE_PATTERN, depende_de_visual_ausente

ROOT = Path(__file__).resolve().parent.parent
SAIDA_PADRAO = ROOT / "outputs" / "calibracao_agentes.json"
DIR_AGENTES = ROOT / "outputs" / "agentes"
CACHE_PADRAO = DIR_AGENTES / "calibracao_cache.jsonl"
LEDGER_PADRAO = DIR_AGENTES / "calibracao_ledger.json"
USO_PADRAO = DIR_AGENTES / "uso_api_calibracao.jsonl"
PROMPTS_DIR = DIR_AGENTES / "calibracao_prompts"
DRYRUN_DIR = DIR_AGENTES / "dryrun" / "calibracao"
TETO_GLOBAL = 160
N_REAL, N_REAL_GEO, SEMENTE = 30, 6, 42
# Limite de aceite do enunciado da tarefa: rejeitar mais de 20% dos reais
# exige ajuste de prompt.
LIMITE_FALSA_REJEICAO = 0.20
# Recalibração de 2026-10-01 (r5+): meta do usuário para a falsa rejeição nos
# reais, DESCONTADOS os itens adjudicados como defeituosos de verdade.
LIMITE_FALSA_REJEICAO_AJUSTADA = 0.15
ADJUDICADOS_DEFEITUOSOS = {
    "REAL-MT5023MH05MT": "gabarito do banco errado: 2.450 - 1.728 = 722 (B); o banco marca A = 732",
    "REAL-MT5060DH20TD": "enunciado sem os dados das três brincadeiras",
}
# Conjunto 3: defeitos conhecidos da auditoria do piloto (adjudicação manual em
# outputs/agentes/piloto_etapa3.json). idx = linha de data/train_curado.jsonl
# (só leitura); o codigo_item confere que a linha não mudou.
DEFEITOS_CONHECIDOS = [
    (1368, "DIST-H15-Fácil-00589", False,
     "hexágono suposto regular: o enunciado não diz que os ângulos são iguais (os dois juízes antigos aprovaram)"),
    (1080, "DIST-H11-Difícil-00301", False,
     "'quatro lados iguais' garante losango, não quadrado; o de 4 m e 4 m também tem os 4 lados iguais "
     "(resposta não única; revisor antigo aprovou)"),
    (1052, "DIST-H03-Difícil-00273", False,
     "C 'Três centenas e sete unidades' também descreve 307 (distrator verdadeiro; revisor antigo aprovou)"),
    (561, "MT2032MH07MT", False, "A (150 + 95) e C (100 + 145) somam 245 (real; os dois juízes reprovaram)"),
    (881, "DIST-H18-Moderado-00102", True,
     "correta: 10 dias a partir de terça terminam na quinta (contagem inclusiva); o revisor antigo deu falso positivo"),
]
# Conjunto de REFERÊNCIA de dificuldade (passo 2). Só itens reais cujo
# codigo_item traz a dificuldade (MT<ano><seq><F|M|D>H<hab><sufixo>) E cujo
# grau_resolucao bate com a letra: 6 itens do banco divergem (ex.: MT2003DH01TD
# rotulado Fácil) e ficam fora — rótulo ambíguo não serve de gabarito.
CODIGO_COM_DIFICULDADE = re.compile(r"^MT\d{4,5}([FMD])H\d+[A-Z]{2}$")
LETRA_DIFICULDADE = {"F": "Fácil", "M": "Moderado", "D": "Difícil"}
N_REF_POR_NIVEL = 15
# Fora da referência: itens com decisão humana (o MT9018MH04MT o usuário
# reclassificou de M para Fácil; os outros 5 reais foram removidos por defeito),
# adjudicados defeituosos e os 4 reais que violam a D5 (pré-filtro P1).
EXCLUIR_REF_DIFICULDADE = {
    "MT9018MH04MT": "decisão humana: rótulo do banco M, reclassificado para Fácil pelo usuário",
    "MT9094DH19MT": "removido por decisão humana", "MT9061FH13MT": "removido por decisão humana",
    "MT2032MH07MT": "removido por decisão humana", "MT9013FH05TD": "removido por decisão humana",
    "MT9049DH10MT": "removido por decisão humana",
    "MT5023MH05MT": "adjudicado defeituoso", "MT5060DH20TD": "adjudicado defeituoso",
    "MT9050MH17TD": "viola a D5 (pré-filtro P1)", "MT9081FH17MT": "viola a D5 (pré-filtro P1)",
    "MT9083MH17MT": "viola a D5 (pré-filtro P1)", "MT9084DH17MT": "viola a D5 (pré-filtro P1)",
}
# Revisão do passo 2 (2026-10-01): os 5 reais que o revisor reprovou na r7
# (contados como "falsa rejeição", 5/44) eram DEFEITOS de verdade — o revisor
# acertou. Três têm a fração trocada por data pela planilha (MT9073FH25TD,
# MT90123MH25MT, MT90125DH25MT; detectados por aq.alternativas_corrompidas_
# planilha, não precisam de lista) e dois têm o gabarito do banco errado,
# conferido por conta exata. Item defeituoso não serve de referência de
# dificuldade (a referência mede itens bons). Saem NO SORTEIO, sem mexer no
# resto: o próximo da mesma fila (nível, ano) entra no lugar, e os outros 40
# julgamentos da r7 continuam valendo (cache).
DEFEITOS_REAIS_CONFIRMADOS = {
    "MT9036DH12TD": "gabarito do banco errado: V = 2·4² − 3·4 + 10 = 32 − 12 + 10 = 30 (D); o banco marca C = 26",
    "MT9009DH03TD": "gabarito do banco errado: (900 − 135) × 1,05 ÷ 4 = 803,25 ÷ 4 = 200,81; o banco marca "
                    "C = R$ 212,25 (a resolução erra a divisão); está em val.jsonl e val_frozen_v1",
}
# Critério da regra D3, FIXADO ANTES de ver os dados do passo 2 (registrado no
# relatório): o revisor sozinho só pode rebaixar se, nos reais M/D da
# referência, rebaixar no máximo 10% (rebaixamento falso) E o kappa ponderado
# quadrático com o banco for >= 0,60. Senão exige o árbitro Gemini (2 de 2),
# que só vale se a regra conjunta cumprir os mesmos 10%; senão a D3 fica
# suspensa (nenhum rebaixamento automático).
LIMITE_REBAIXAMENTO_FALSO = 0.10
KAPPA_MIN_D3 = 0.60
D5_PADRAO = ROOT / "data" / "calibracao_d5.json"
TRAIN_CURADO = ROOT / "data" / "train_curado.jsonl"
INJECAO = ROOT / "data" / "injecao_saeb.jsonl"
# Aceitas que SAÍRAM da injeção (auditoria independente, rejulgamento) ficam
# aqui com o exemplo inteiro (campo "exemplo"): D5-B1/B2 foram movidos na
# revisão do piloto 2 e continuam valendo como cópia literal.
REJEITADAS_INJECAO = ROOT / "outputs" / "injecao_rejeitadas.jsonl"

# Geometria plana com classificação/propriedade de figura: é onde o validador
# pode sobrecorrigir (exigir critério explícito, ver hierarquia onde não há).
GEO = re.compile(r"tri[aâ]ngulo|quadril[aá]tero|losango|trap[eé]zio|paralelogramo|ret[aâ]ngulo|quadrado",
                 re.I)
# Remissões a algo que só existe na prova impressa. "Observe" sozinho não
# entra: vários itens do banco dizem "Observe quadrilátero com..." e dão os
# dados no próprio texto.
REMISSAO_VISUAL = re.compile(r"\b(abaixo|ao lado|a seguir|na figura|no gr[aá]fico|na tabela|no quadro|"
                             r"na malha|na imagem|no desenho|representad[oa] (?:na|no|abaixo))\b", re.I)


# ---------------------------------------------------------------------------
# Conjuntos
# ---------------------------------------------------------------------------

def carregar_cal20():
    itens = ab.carregar_calibracao()
    for it in itens:
        it["conjunto"] = "cal20"
    return itens


def _elegivel(row):
    """(questao, meta) do item real ou (None, motivo). Mesmo conversor do treino
    (build_example), mais exclusões de dependência visual que o treino não faz
    porque lá a figura não é julgada por ninguém."""
    ex, motivo = build_example(row)
    if ex is None:
        return None, motivo
    keys = row.keys()
    if "depende_de_imagem" in keys and row["depende_de_imagem"]:
        return None, "depende_de_imagem"
    for col in ("descricao_imagem", "imagem_path"):
        if col in keys and str(row[col] or "").strip():
            return None, "descricao_imagem"
    q = json.loads(ex["messages"][2]["content"])["questoes"][0]
    texto = q["enunciado"] + " " + " ".join(q["alternativas"].values())
    if depende_de_visual_ausente(texto) or IMAGE_PATTERN.search(texto) or REMISSAO_VISUAL.search(texto):
        return None, "remissao_visual"
    if len(q["resolucao_passo_a_passo"].strip()) < 20:
        return None, "sem_resolucao"
    return (q, ex), None


def carregar_real(n=N_REAL, n_geo=N_REAL_GEO, seed=SEMENTE, rows=None):
    """n itens reais: n_geo de geometria plana + o resto estratificado por ano
    (round-robin entre anos; dentro do ano, habilidades diferentes primeiro)."""
    rows = rows if rows is not None else load_rows()
    pool, motivos = [], Counter()
    for r in rows:
        par, motivo = _elegivel(r)
        if par is None:
            motivos[motivo] += 1
            continue
        q, ex = par
        m = ex["meta"]
        user = ex["messages"][1]["content"]
        desc = re.search(r"Habilidade: .*? — (.*)\. Dificuldade:", user)
        pool.append({"id": f"REAL-{m['codigo_item']}", "conjunto": "real30", "ano": m["ano"],
                     "habilidade": m["habilidade"], "descricao": desc.group(1) if desc else "",
                     "dificuldade": m.get("dificuldade") or "Moderado", "questao": q, "boa": True,
                     "codigo_item": m["codigo_item"], "lote": m.get("lote") or "original"})
    rng = random.Random(seed)
    pool.sort(key=lambda it: it["id"])
    geo = [it for it in pool if GEO.search(it["questao"]["enunciado"] + " "
                                           + " ".join(it["questao"]["alternativas"].values()))]
    rng.shuffle(geo)
    escolhidos = geo[:n_geo]
    ids = {it["id"] for it in escolhidos}
    por_ano = {}
    for it in pool:
        if it["id"] not in ids:
            por_ano.setdefault(it["ano"], []).append(it)
    filas = {}
    for ano, lst in sorted(por_ano.items()):
        rng.shuffle(lst)
        # habilidades distintas primeiro: mais cobertura com o mesmo custo
        vistas, primeiro, depois = set(), [], []
        for it in lst:
            (depois if it["habilidade"] in vistas else primeiro).append(it)
            vistas.add(it["habilidade"])
        filas[ano] = primeiro + depois
    while len(escolhidos) < n and any(filas.values()):
        for ano in sorted(filas):
            if filas[ano] and len(escolhidos) < n:
                escolhidos.append(filas[ano].pop(0))
    return escolhidos, {"elegiveis": len(pool), "excluidos": dict(motivos), "geometria_elegiveis": len(geo)}


def _defeito_no_sorteio(it, defeitos):
    """Motivo para descartar um item JÁ sorteado (None se é bom)."""
    if it["codigo_item"] in defeitos:
        return defeitos[it["codigo_item"]]
    datas = aq.alternativas_corrompidas_planilha(it["questao"])
    if datas:
        return f"alternativas com data da planilha no lugar da fração ({', '.join(sorted(datas))})"
    return None


def carregar_ref_dificuldade(n_por_nivel=N_REF_POR_NIVEL, seed=SEMENTE, rows=None, excluir=None, defeitos=None):
    """Referência de dificuldade (passo 2): n_por_nivel itens reais de cada
    nível (F/M/D), rodízio entre anos e, dentro do ano, habilidades diferentes
    primeiro. O rótulo vem do banco (letra do codigo_item = grau_resolucao).
    Mesma elegibilidade do real30 (sem imagem/remissão visual), porque o
    revisor só vê o texto.

    `defeitos` ({codigo: motivo}, padrão DEFEITOS_REAIS_CONFIRMADOS) e as
    alternativas com data da planilha saem DEPOIS do embaralhamento: o item
    defeituoso é pulado e o próximo da fila entra; os demais sorteados não
    mudam (o `excluir`, ao contrário, sai antes e muda o sorteio todo)."""
    excluir = EXCLUIR_REF_DIFICULDADE if excluir is None else excluir
    defeitos = DEFEITOS_REAIS_CONFIRMADOS if defeitos is None else defeitos
    descartados = {}
    rows = rows if rows is not None else load_rows()
    pool, motivos = {}, Counter()
    for r in rows:
        cod = str(r["codigo_item"] or "")
        m = CODIGO_COM_DIFICULDADE.match(cod)
        if not m:
            motivos["codigo_sem_dificuldade"] += 1
            continue
        if cod in excluir:
            motivos["excluido_lista"] += 1
            continue
        par, motivo = _elegivel(r)
        if par is None:
            motivos[motivo] += 1
            continue
        q, ex = par
        meta = ex["meta"]
        rotulo = LETRA_DIFICULDADE[m.group(1)]
        if meta.get("dificuldade") != rotulo:
            motivos["letra_diverge_do_grau"] += 1
            continue
        user = ex["messages"][1]["content"]
        desc = re.search(r"Habilidade: .*? — (.*)\. Dificuldade:", user)
        pool.setdefault(rotulo, {}).setdefault(meta["ano"], []).append(
            {"id": f"DIF-{cod}", "conjunto": "dif45", "ano": meta["ano"], "habilidade": meta["habilidade"],
             "descricao": desc.group(1) if desc else "", "dificuldade": rotulo, "questao": q, "boa": True,
             "codigo_item": cod})
    rng = random.Random(seed)
    escolhidos = []
    for nivel in ("Fácil", "Moderado", "Difícil"):
        filas = {}
        for ano, lst in sorted(pool.get(nivel, {}).items()):
            lst.sort(key=lambda it: it["id"])
            rng.shuffle(lst)
            vistas, primeiro, depois = set(), [], []
            for it in lst:
                (depois if it["habilidade"] in vistas else primeiro).append(it)
                vistas.add(it["habilidade"])
            filas[ano] = primeiro + depois
        n = 0
        while n < n_por_nivel and any(filas.values()):
            for ano in sorted(filas):
                while filas[ano] and n < n_por_nivel:
                    it = filas[ano].pop(0)
                    motivo = _defeito_no_sorteio(it, defeitos)
                    if motivo:
                        descartados[it["codigo_item"]] = motivo
                        continue
                    escolhidos.append(it)
                    n += 1
                    break
    elegiveis = {nivel: {ano: len(v) for ano, v in d.items()} for nivel, d in pool.items()}
    return escolhidos, {"elegiveis": elegiveis, "excluidos": dict(motivos),
                        "excluidos_lista": sorted(excluir), "descartados_defeito": descartados}


def carregar_defeitos(path=TRAIN_CURADO, casos=DEFEITOS_CONHECIDOS):
    """Conjunto 3 (defeitos conhecidos). Lê data/train_curado.jsonl SÓ para
    leitura e confere o codigo_item de cada idx."""
    alvo = {idx for idx, *_ in casos}
    linhas = {}
    with open(path, encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if i in alvo:
                linhas[i] = json.loads(linha)
    itens = []
    for idx, codigo, boa, motivo in casos:
        ex = linhas.get(idx)
        if ex is None or ex["meta"].get("codigo_item") != codigo:
            raise ValueError(f"idx {idx}: esperado {codigo}, achado "
                             f"{None if ex is None else ex['meta'].get('codigo_item')}")
        m = ex["meta"]
        itens.append({"id": f"DEF-{idx}", "conjunto": "defeitos", "ano": m["ano"], "habilidade": m["habilidade"],
                      "descricao": ab.descricao_do_prompt(ex), "dificuldade": m.get("dificuldade"),
                      "questao": ab.questoes_do_exemplo(ex)[0], "boa": boa, "codigo_item": codigo,
                      "motivo_rotulo": motivo})
    return itens


def carregar_d5(path=D5_PADRAO, injecao=INJECAO, rejeitadas=REJEITADAS_INJECAO):
    """Conjunto 4 (D5): ruins com uma alternativa errada verdadeira por outro
    eixo e controles corretos. Os itens com "copiar_de" vêm literalmente da
    injeção do piloto (data/injecao_saeb.jsonl) ou, se já saíram dela, do
    exemplo guardado em outputs/injecao_rejeitadas.jsonl."""
    dados = json.loads(Path(path).read_text(encoding="utf-8"))
    copias = {c["copiar_de"] for c in dados["itens"] if c.get("copiar_de")}
    achados = {}
    if copias:
        for arq, campo in ((rejeitadas, "exemplo"), (injecao, None)):  # a injeção vence
            if not Path(arq).exists():
                continue
            with open(arq, encoding="utf-8") as f:
                for linha in f:
                    try:
                        ex = json.loads(linha)
                    except json.JSONDecodeError:
                        continue
                    ex = ex.get(campo) if campo else ex
                    if isinstance(ex, dict) and ex.get("meta", {}).get("codigo_item") in copias:
                        achados[ex["meta"]["codigo_item"]] = ex
    descricoes = _descricoes()
    itens = []
    for c in dados["itens"]:
        if c.get("copiar_de"):
            ex = achados.get(c["copiar_de"])
            if ex is None:
                raise ValueError(f"{c['id']}: {c['copiar_de']} não está em {injecao}")
            m = ex["meta"]
            it = {"ano": m["ano"], "habilidade": m["habilidade"], "descricao": ab.descricao_do_prompt(ex),
                  "dificuldade": m.get("dificuldade"), "questao": ab.questoes_do_exemplo(ex)[0],
                  "codigo_item": c["copiar_de"]}
        else:
            it = {k: c[k] for k in ("ano", "habilidade", "dificuldade", "questao")}
            it["descricao"] = c.get("descricao") or descricoes.get((c["ano"], c["habilidade"]), "")
        it.update({"id": c["id"], "conjunto": "d5", "boa": bool(c["boa"]), "motivo_rotulo": c.get("verificacao")})
        itens.append(it)
    return itens


def _descricoes():
    """{(ano, habilidade): descrição} do prompt de treino (para os itens D5
    escritos à mão receberem a MESMA descrição que a injeção usaria)."""
    out = {}
    try:
        with open(TRAIN_CURADO, encoding="utf-8") as f:
            for linha in f:
                ex = json.loads(linha)
                m = ex["meta"]
                out.setdefault((m["ano"], m["habilidade"]), ab.descricao_do_prompt(ex))
    except (OSError, json.JSONDecodeError):
        pass
    return out


# ---------------------------------------------------------------------------
# Agentes com cache e orçamento global persistente
# ---------------------------------------------------------------------------

class Ledger:
    """Total de chamadas REAIS (e tokens) de todas as execuções da calibração.
    Gravado a cada chamada: uma queda no meio não "devolve" orçamento."""

    def __init__(self, path, teto):
        self.path = Path(path)
        self.teto = int(teto)
        self.d = {"chamadas": 0, "prompt_tokens": 0, "completion_tokens": 0, "por_papel": {},
                  "por_rodada": {}}
        if self.path.exists():
            self.d.update(json.loads(self.path.read_text(encoding="utf-8")))

    def restante(self):
        return max(0, self.teto - self.d["chamadas"])

    def somar(self, rodada, papel, chamadas, pt, ct):
        for alvo in (self.d, self.d["por_papel"].setdefault(papel, {}),
                     self.d["por_rodada"].setdefault(rodada, {})):
            for k, v in (("chamadas", chamadas), ("prompt_tokens", pt), ("completion_tokens", ct)):
                alvo[k] = alvo.get(k, 0) + v
        aq._garantir_gravavel(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=2), encoding="utf-8")


def _chave_cache(papel, modelo, mensagens):
    # Mesma chave usada pela auditoria (aq.CacheRespostas): a auditoria
    # reaproveita as respostas da calibração quando a mensagem é idêntica.
    return aq.chave_cache(papel, modelo, mensagens)


class AgentesComCache(aq.Agentes):
    """Agentes que consultam o cache antes de gastar. Só respostas que
    chegaram (texto != None) entram no cache; erro de API é refeito."""

    def __init__(self, *a, cache_path, ledger, rodada, **kw):
        super().__init__(*a, **kw)
        self.cache_path = Path(cache_path)
        self.ledger = ledger
        self.rodada = rodada
        self.hits = 0
        self.cache = {}
        if self.cache_path.exists():
            for linha in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue  # linha final truncada por queda
                self.cache[r["chave"]] = r

    def chamar(self, papel, fase, mensagens):
        modelo = self.modelos[papel]
        chave = _chave_cache(papel, modelo, mensagens)
        # Revisão do piloto 2: resposta malformada/truncada não vale como cache
        # (nem entra nele): reprova naquela rodada, mas é refeita na próxima.
        if chave in self.cache and aq.resposta_interpretavel(papel, self.cache[chave].get("texto")):
            self.hits += 1
            r = self.cache[chave]
            return r["texto"], {"cache": True, "prompt_tokens": r.get("prompt_tokens"),
                                "completion_tokens": r.get("completion_tokens")}
        antes = (self.orcamento.chamadas, self.orcamento.prompt_tokens, self.orcamento.completion_tokens)
        try:
            texto, info = super().chamar(papel, fase, mensagens)
        finally:
            d = (self.orcamento.chamadas - antes[0], self.orcamento.prompt_tokens - antes[1],
                 self.orcamento.completion_tokens - antes[2])
            if any(d):
                self.ledger.somar(self.rodada, papel, *d)
        # P4 (2026-10-01): resposta cortada pelo max_tokens (finish_reason
        # "length") também não entra, mesmo que o JSON final feche.
        if texto is not None and not aq.resposta_truncada(info) and (
                papel not in aq.PAPEIS_CACHEAVEIS or aq.resposta_interpretavel(papel, texto)):
            reg = {"chave": chave, "papel": papel, "modelo": modelo, "texto": texto,
                   "prompt_tokens": info.get("prompt_tokens"), "completion_tokens": info.get("completion_tokens"),
                   "ts": aq.agora_iso()}
            self.cache[chave] = reg
            with aq.abrir_para_gravar(self.cache_path, "a") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        return texto, info


def montar(args, rodada, caminhos_):
    ledger = Ledger(caminhos_["ledger"], args.teto_global)
    teto = ledger.restante() if args.max_chamadas is None else min(args.max_chamadas, ledger.restante())
    base = aq.montar_agentes(args, simulado=args.dry_run, log_uso=caminhos_["uso"],
                             orcamento=aq.Orcamento(teto, args.max_tokens_total))
    return AgentesComCache(base.clientes, base.modelos, base.orcamento, log_uso=caminhos_["uso"],
                           tentativas=args.tentativas_api, backoff=args.backoff, simulado=args.dry_run,
                           cache_path=caminhos_["cache"], ledger=ledger, rodada=rodada)


# ---------------------------------------------------------------------------
# Execução e métricas
# ---------------------------------------------------------------------------

def _juiz(j):
    if j is None:
        return None
    out = {"veredito": bool(j.get("veredito")), "avaliado": bool(j.get("avaliado")),
           "resposta": j.get("resposta_calculada"), "problemas": ab._codigos(j),
           "detalhes": [p.get("detalhe", "")[:200] for p in j.get("problemas") or []][:6]}
    for k in ("fases", "criterios_falhos", "criterios_bloqueantes_falhos", "avisos", "status", "erro", "confianca",
              "dificuldade_real"):
        if k in j:
            out[k] = j[k]
    return out


def julgar_item(agentes, it):
    if it["conjunto"] == "dif45":
        # Referência de dificuldade: só o revisor (a dificuldade_real é dele).
        rev = agentes.revisar(it["questao"], it["ano"], it["habilidade"], it["descricao"], None, it["dificuldade"])
        return {"id": it["id"], "conjunto": it["conjunto"], "boa": it["boa"], "ano": it["ano"],
                "habilidade": it["habilidade"], "gabarito": it["questao"].get("resposta_correta"),
                "dificuldade_rotulo": it.get("dificuldade"), "codigo_item": it.get("codigo_item"),
                "validador": None, "revisor": _juiz(rev), "par": None}
    j = agentes.julgar(it["questao"], it["ano"], it["habilidade"], it["descricao"], None,
                       it["dificuldade"], curto_circuito=False)
    linha = {"id": it["id"], "conjunto": it["conjunto"], "boa": it["boa"], "ano": it["ano"],
             "habilidade": it["habilidade"], "gabarito": it["questao"].get("resposta_correta"),
             "dificuldade_rotulo": it.get("dificuldade"),
             "validador": _juiz(j["validador"]), "revisor": _juiz(j["revisor"]), "par": bool(j["ambos"])}
    for k in ("rotulo_log", "motivo_rotulo", "codigo_item"):
        if k in it:
            linha[k] = it[k]
    return linha


def matriz(linhas, campo):
    """Classe positiva = APROVAR. vp: boa aprovada; fn: boa reprovada;
    fp: ruim aprovada; vn: ruim reprovada. Não avaliado conta como reprovação
    (fail-closed) e é contado à parte."""
    def ok(l):
        return l["par"] if campo == "par" else l[campo]["veredito"]
    m = {"vp": 0, "fn": 0, "fp": 0, "vn": 0}
    for l in linhas:
        m[("vp" if ok(l) else "fn") if l["boa"] else ("fp" if ok(l) else "vn")] += 1
    boas, ruins = m["vp"] + m["fn"], m["fp"] + m["vn"]
    m["aprova_boas"] = f"{m['vp']}/{boas}"
    m["reprova_ruins"] = f"{m['vn']}/{ruins}"
    m["ic95_reprova_ruins"] = ab.wilson(m["vn"], ruins) if ruins else None
    m["ic95_aprova_boas"] = ab.wilson(m["vp"], boas) if boas else None
    if campo != "par":
        m["nao_avaliados"] = sum(not l[campo]["avaliado"] for l in linhas)
    return m


def itens_da_rodada(saida, nome):
    """{id: linha} da rodada, incluindo os itens herdados de outra rodada."""
    rod = saida["rodadas"][nome]
    out = {}
    if rod.get("herda_de"):
        out.update({i: dict(l, herdado_de=l.get("herdado_de") or rod["herda_de"])
                    for i, l in itens_da_rodada(saida, rod["herda_de"]).items()})
    out.update(rod["itens"])
    return out


def revisar_de_novo(agentes, it, anterior):
    """Refaz SÓ o revisor (mudança na regra de decisão do código, prompt igual:
    a resposta vem do cache) e mantém o validador da rodada anterior."""
    rev = agentes.revisar(it["questao"], it["ano"], it["habilidade"], it["descricao"], None, it["dificuldade"])
    linha = {k: v for k, v in anterior.items() if k != "herdado_de"}
    linha["validador_de"] = anterior.get("herdado_de") or anterior.get("versao_prompts")
    linha["revisor"] = _juiz(rev)
    linha["par"] = bool(linha["validador"]["veredito"] and rev["veredito"])
    return linha


def metricas_conjunto(linhas):
    """Matriz (boa/ruim) de cada juiz e do par num conjunto rotulado, com os
    ids que erraram (ruim aceita / boa rejeitada)."""
    out = {c: matriz(linhas, c) for c in ("validador", "revisor", "par")}
    for c in ("validador", "revisor", "par"):
        ok = (lambda l: l["par"]) if c == "par" else (lambda l, c=c: l[c]["veredito"])
        out[c]["ruins_aceitas"] = sorted(l["id"] for l in linhas if not l["boa"] and ok(l))
        out[c]["boas_rejeitadas"] = sorted(l["id"] for l in linhas if l["boa"] and not ok(l))
    out["kappa_validador_revisor"] = ab.kappa([(l["validador"]["veredito"], l["revisor"]["veredito"])
                                               for l in linhas])
    out["n"] = len(linhas)
    return out


def metricas_geral(linhas):
    """Conjunto TODO: kappa validador x revisor, taxa de reprovação de cada
    juiz e do par, e não avaliados (fail-closed)."""
    n = len(linhas)
    if not n:
        return {}
    out = {"n": n, "kappa_validador_revisor": ab.kappa(
        [(l["validador"]["veredito"], l["revisor"]["veredito"]) for l in linhas])}
    for c in ("validador", "revisor", "par"):
        rej = sum(not (l["par"] if c == "par" else l[c]["veredito"]) for l in linhas)
        out[f"reprova_{c}"] = {"n": rej, "taxa": round(rej / n, 3), "ic95": ab.wilson(rej, n)}
        if c != "par":
            out[f"nao_avaliados_{c}"] = sorted(l["id"] for l in linhas if not l[c]["avaliado"])
    v = [l["validador"]["veredito"] for l in linhas]
    r = [l["revisor"]["veredito"] for l in linhas]
    out["discordancias"] = {"so_validador_reprova": sorted(l["id"] for l, a, b in zip(linhas, v, r) if not a and b),
                            "so_revisor_reprova": sorted(l["id"] for l, a, b in zip(linhas, v, r) if a and not b)}
    return out


def filtro_deterministico(questao, ano=None, habilidade=None):
    """Filtros EXATOS que a injeção aplica ANTES dos juízes (filtro_injecao) e
    que o rejulgamento/montagem aplicam às injetadas (0 chamadas): verificador
    aritmético que contradiz o gabarito, verificador de geometria que reprova e
    o pré-filtro D5 de geometria (P1). Lista de motivos ([] = passa)."""
    out = []
    if aq.verificar_aritmetica(questao)["status"] == "contradiz":
        out.append("verificador_aritmetico")
    if aq.veredito_geometria(questao, ano, habilidade)["status"] == "reprovada":
        out.append("geometria")
    if aq.veredito_d5_geometria(questao)["status"] == "reprovada":
        out.append("geometria_d5")
    return out


def metricas_pipeline(linhas):
    """Matriz do PIPELINE (filtro determinístico + par): é o que decide a
    entrada na injeção. Só linhas anotadas com filtro_deterministico."""
    anot = [l for l in linhas if "filtro_deterministico" in l and l.get("par") is not None]
    if not anot:
        return None
    m = {"vp": 0, "fn": 0, "fp": 0, "vn": 0}
    ruins_aceitas, boas_rej, barradas = [], [], []
    for l in anot:
        ok = l["par"] and not l["filtro_deterministico"]
        if l["filtro_deterministico"]:
            barradas.append(l["id"])
        if l["boa"]:
            m["vp" if ok else "fn"] += 1
            if not ok:
                boas_rej.append(l["id"])
        else:
            m["fp" if ok else "vn"] += 1
            if ok:
                ruins_aceitas.append(l["id"])
    return dict(m, n=len(anot), ruins_aceitas=sorted(ruins_aceitas), boas_rejeitadas=sorted(boas_rej),
                barradas_pelo_filtro=sorted(barradas))


def metricas_r5(linhas):
    """Metas da recalibração de 2026-10-01 (D1-D5) sobre os quatro conjuntos."""
    por = {}
    for l in linhas:
        por.setdefault(l["conjunto"], []).append(l)
    out, falhas = {}, []
    if por.get("cal20"):
        cal = por["cal20"]
        out["cal20_rotulo_d5"] = metricas_conjunto(cal)
        if any("rotulo_log" in l for l in cal):
            out["cal20_rotulo_log"] = metricas_conjunto([dict(l, boa=l.get("rotulo_log", l["boa"])) for l in cal])
        p = out["cal20_rotulo_d5"]["par"]
        if p["fp"]:
            falhas.append(f"cal20: par aceitou ruins {p['ruins_aceitas']}")
        if p["fn"]:
            falhas.append(f"cal20: par rejeitou boas {p['boas_rejeitadas']}")
    if por.get("real30"):
        reais = por["real30"]
        rej = [l["id"] for l in reais if not l["par"]]
        ajust = [i for i in rej if i not in ADJUDICADOS_DEFEITUOSOS]
        n_aj = len([l for l in reais if l["id"] not in ADJUDICADOS_DEFEITUOSOS])
        out["real_falsa_rejeicao"] = {
            "bruta": f"{len(rej)}/{len(reais)}", "taxa_bruta": round(len(rej) / len(reais), 3),
            "ajustada": f"{len(ajust)}/{n_aj}", "taxa_ajustada": round(len(ajust) / n_aj, 3) if n_aj else None,
            "ic95_ajustada": ab.wilson(len(ajust), n_aj) if n_aj else None,
            "adjudicados_descontados": sorted(ADJUDICADOS_DEFEITUOSOS),
            "adjudicados_reprovados": sorted(i for i in rej if i in ADJUDICADOS_DEFEITUOSOS),
            "rejeitados_ajustados": ajust}
        if n_aj and len(ajust) / n_aj > LIMITE_FALSA_REJEICAO_AJUSTADA:
            falhas.append(f"reais: falsa rejeição ajustada {len(ajust)}/{n_aj} > "
                          f"{LIMITE_FALSA_REJEICAO_AJUSTADA:.0%}")
    for nome in ("defeitos", "d5"):
        if por.get(nome):
            m = metricas_conjunto(por[nome])
            out[nome] = m
            if m["par"]["fp"]:
                falhas.append(f"{nome}: par aceitou ruins {m['par']['ruins_aceitas']}")
            if m["par"]["fn"]:
                falhas.append(f"{nome}: par rejeitou boas {m['par']['boas_rejeitadas']}")
    if por.get("defeitos"):
        # D1: o teste que importa é o REVISOR sozinho (antes era carimbo).
        m = out["defeitos"]["revisor"]
        if m["fp"] or m["fn"]:
            falhas.append(f"defeitos: revisor sozinho aceitou {m['ruins_aceitas']} / rejeitou {m['boas_rejeitadas']}")
    out["geral"] = metricas_geral(linhas)
    pip = metricas_pipeline(linhas)
    if pip:
        out["pipeline_filtro_mais_par"] = pip
    out["metas_falhas"] = falhas
    out["metas_ok"] = not falhas
    out["criterio"] = ("cal20 (rótulos D5): par reprova todas as ruins e aprova todas as boas; reais: falsa rejeição "
                       f"do par <= {LIMITE_FALSA_REJEICAO_AJUSTADA:.0%} descontados os adjudicados; defeitos: par e "
                       "revisor sozinho reprovam os defeituosos e aprovam o correto; d5: par reprova as 6 ruins e "
                       "aprova os 6 controles")
    return out


NIVEIS = ("Fácil", "Moderado", "Difícil")


def kappa_ponderado(pares, pesos="quadratico", niveis=NIVEIS):
    """Kappa de Cohen ponderado para rótulos ORDINAIS (Fácil < Moderado <
    Difícil): errar por dois níveis pesa mais que por um. pesos "linear"
    (|i-j|/(k-1)), "quadratico" ((i-j)²/(k-1)²) ou "nominal" (kappa simples:
    qualquer discordância pesa 1). None se indefinido."""
    pares = [(a, b) for a, b in pares if a in niveis and b in niveis]
    n, k = len(pares), len(niveis)
    if not n:
        return None
    ix = {v: i for i, v in enumerate(niveis)}

    def w(i, j):
        if pesos == "nominal":
            return float(i != j)
        d = abs(i - j) / (k - 1)
        return d if pesos == "linear" else d * d
    obs = Counter((ix[a], ix[b]) for a, b in pares)
    ma = Counter(ix[a] for a, _ in pares)
    mb = Counter(ix[b] for _, b in pares)
    do = sum(w(i, j) * c for (i, j), c in obs.items()) / n
    de = sum(w(i, j) * ma[i] * mb[j] for i in range(k) for j in range(k)) / (n * n)
    return None if de == 0 else round(1 - do / de, 3)


def _matriz_dif(pares):
    """{rótulo_banco: {julgado: n}} nos três níveis (linhas = banco)."""
    return {a: {b: sum(1 for x, y in pares if x == a and y == b) for b in NIVEIS} for a in NIVEIS}


def _concordancia(pares):
    pares = [(a, b) for a, b in pares if a in NIVEIS and b in NIVEIS]
    n = len(pares)
    md = [(a, b) for a, b in pares if a in ("Moderado", "Difícil")]
    reb = sum(1 for a, b in md if b == "Fácil")
    out = {"n": n, "matriz_banco_x_julgado": _matriz_dif(pares),
           "exata": round(sum(a == b for a, b in pares) / n, 3) if n else None,
           "kappa_ponderado_quadratico": kappa_ponderado(pares, "quadratico"),
           "kappa_ponderado_linear": kappa_ponderado(pares, "linear"),
           "kappa_simples": kappa_ponderado(pares, "nominal"),
           "rebaixaria_para_facil": {"n": reb, "de": len(md), "taxa": round(reb / len(md), 3) if md else None,
                                     "ic95": ab.wilson(reb, len(md)) if md else None},
           "mais_facil_que_o_banco": sum(NIVEIS.index(b) < NIVEIS.index(a) for a, b in pares),
           "mais_dificil_que_o_banco": sum(NIVEIS.index(b) > NIVEIS.index(a) for a, b in pares)}
    return out


def metricas_dificuldade(linhas, selecao=None):
    """Passo 2: concordância da dificuldade_real (revisor sem âncora e, onde
    houver, árbitro Gemini) com o rótulo do banco no conjunto dif45, mais a
    regra "2 de 2" (os dois dizem Fácil) sobre os M/D que o Gemini julgou.

    `selecao` ({"ids", "descartados_defeito"} de carregar_ref_dificuldade):
    só as linhas da seleção ATUAL entram nas métricas. As que ficaram de fora
    por defeito confirmado e que o revisor reprovou são ACERTO dele
    (revisor_rejeicao_correta_defeitos), não falsa rejeição — na r7 os 5
    "falsos" 5/44 eram todos defeitos de verdade."""
    todas = [l for l in linhas if l["conjunto"] == "dif45"]
    fora, corretas = [], []
    if selecao and selecao.get("ids"):
        sel = set(selecao["ids"])
        defe = selecao.get("descartados_defeito") or {}
        fora = sorted(l["id"] for l in todas if l["id"] not in sel)
        corretas = sorted(l["id"] for l in todas if l["id"] not in sel and l.get("codigo_item") in defe
                          and (l.get("revisor") or {}).get("avaliado") and not l["revisor"]["veredito"])
        todas = [l for l in todas if l["id"] in sel]
    dif = [l for l in todas if (l.get("revisor") or {}).get("dificuldade_real")]
    if not dif:
        return {}
    pares_rev = [(l["dificuldade_rotulo"], l["revisor"]["dificuldade_real"]) for l in dif]
    out = {"revisor": _concordancia(pares_rev)}
    out["revisor"]["por_ano"] = {ano: _concordancia([(l["dificuldade_rotulo"], l["revisor"]["dificuldade_real"])
                                                     for l in dif if l["ano"] == ano])
                                 for ano in sorted({l["ano"] for l in dif})}
    for k in ("matriz_banco_x_julgado",):
        for v in out["revisor"]["por_ano"].values():
            v.pop(k, None)
    out["revisor"]["rebaixaria_ids"] = sorted(l["id"] for l in dif if l["dificuldade_rotulo"] != "Fácil"
                                              and l["revisor"]["dificuldade_real"] == "Fácil")
    # veredito do revisor sozinho nesses reais (informativo: falsa rejeição)
    rej = sorted(l["id"] for l in dif if not l["revisor"]["veredito"])
    out["revisor_falsa_rejeicao_reais"] = {"n": len(rej), "de": len(dif), "taxa": round(len(rej) / len(dif), 3),
                                           "ic95": ab.wilson(len(rej), len(dif)), "ids": rej}
    if selecao and selecao.get("ids"):
        out["fora_da_selecao"] = fora
        out["revisor_rejeicao_correta_defeitos"] = {"n": len(corretas), "ids": corretas,
                                                    "motivos": {i: (selecao.get("descartados_defeito") or {})
                                                                .get(i[4:]) for i in corretas}}
        out["sem_dificuldade_real"] = sorted(l["id"] for l in todas if l not in dif)
    gem = [l for l in dif if (l.get("gemini") or {}).get("avaliado")]
    if gem:
        out["gemini"] = _concordancia([(l["dificuldade_rotulo"], l["gemini"]["dificuldade_real"]) for l in gem])
        out["gemini"]["nota"] = ("amostra do Gemini: todos os M/D da referência e uma amostra de F (orçamento de "
                                 "40 chamadas); a matriz não é estratificada como a do revisor")
        out["revisor_no_subconjunto_gemini"] = _concordancia([(l["dificuldade_rotulo"],
                                                               l["revisor"]["dificuldade_real"]) for l in gem])
        out["revisor_x_gemini_kappa_quadratico"] = kappa_ponderado(
            [(l["revisor"]["dificuldade_real"], l["gemini"]["dificuldade_real"]) for l in gem])
        md = [l for l in gem if l["dificuldade_rotulo"] in ("Moderado", "Difícil")]
        dois = sorted(l["id"] for l in md if l["revisor"]["dificuldade_real"] == "Fácil"
                      and l["gemini"]["dificuldade_real"] == "Fácil")
        so_rev = [l for l in md if l["revisor"]["dificuldade_real"] == "Fácil"]
        out["regra_2_de_2"] = {"rebaixaria": len(dois), "de": len(md),
                               "taxa": round(len(dois) / len(md), 3) if md else None,
                               "ic95": ab.wilson(len(dois), len(md)) if md else None, "ids": dois,
                               "revisor_facil_gemini_discorda": len(so_rev) - len(dois)}
        fac = [l for l in gem if l["dificuldade_rotulo"] == "Fácil"]
        out["regra_2_de_2"]["facil_confirmado_nos_F"] = (
            f"{sum(l['revisor']['dificuldade_real'] == 'Fácil' and l['gemini']['dificuldade_real'] == 'Fácil' for l in fac)}"
            f"/{len(fac)}")
    out["regra_d3"] = regra_d3(out)
    return out


def regra_d3(md):
    """Regra D3 que os números sustentam (critério fixado antes dos dados,
    LIMITE_REBAIXAMENTO_FALSO e KAPPA_MIN_D3):
      "revisor"            — o revisor sozinho rebaixa;
      "revisor_e_arbitro"  — rebaixa só com o revisor E o Gemini dizendo Fácil;
      "suspensa"           — nenhum rebaixamento automático.
    Rebaixamento falso = item real M/D (rótulo do banco) julgado Fácil."""
    rev = md.get("revisor") or {}
    taxa_rev = (rev.get("rebaixaria_para_facil") or {}).get("taxa")
    kq = rev.get("kappa_ponderado_quadratico")
    out = {"limite_rebaixamento_falso": LIMITE_REBAIXAMENTO_FALSO, "kappa_min": KAPPA_MIN_D3,
           "revisor_taxa": taxa_rev, "revisor_kappa_quadratico": kq}
    if taxa_rev is None:
        return dict(out, regra="suspensa", motivo="sem dados do revisor")
    if taxa_rev <= LIMITE_REBAIXAMENTO_FALSO and kq is not None and kq >= KAPPA_MIN_D3:
        return dict(out, regra="revisor", motivo="o revisor sozinho cumpre o limite de rebaixamento falso e o kappa")
    dois = md.get("regra_2_de_2")
    if not dois or dois.get("taxa") is None:
        return dict(out, regra="suspensa", motivo="revisor fora do limite e sem medida do árbitro")
    out["dois_de_dois_taxa"] = dois["taxa"]
    out["dois_de_dois_ic95"] = dois.get("ic95")
    if dois["taxa"] <= LIMITE_REBAIXAMENTO_FALSO:
        return dict(out, regra="revisor_e_arbitro",
                    motivo="o revisor sozinho rebaixa demais; com o Gemini confirmando (2 de 2) fica no limite")
    return dict(out, regra="suspensa", motivo="nem o revisor sozinho nem o 2 de 2 cumprem o limite")


def metricas(linhas_cal, linhas_real):
    out = {}
    if linhas_cal:
        out["cal20"] = {c: matriz(linhas_cal, c) for c in ("validador", "revisor", "par")}
        ruins = [l for l in linhas_cal if not l["boa"]]
        out["cal20"]["ruins_aceitas_pelo_par"] = [l["id"] for l in ruins if l["par"]]
        out["cal20"]["boas_rejeitadas_pelo_par"] = [l["id"] for l in linhas_cal if l["boa"] and not l["par"]]
        out["cal20"]["kappa_validador_revisor"] = ab.kappa(
            [(l["validador"]["veredito"], l["revisor"]["veredito"]) for l in linhas_cal])
        out["cal20"]["n"] = len(linhas_cal)
    if linhas_real:
        n = len(linhas_real)
        r = {"n": n}
        for c in ("validador", "revisor", "par"):
            rej = [l["id"] for l in linhas_real if not (l["par"] if c == "par" else l[c]["veredito"])]
            r[c] = {"rejeitados": len(rej), "taxa_falsa_rejeicao": round(len(rej) / n, 3),
                    "ic95": ab.wilson(len(rej), n), "ids": rej}
            if c != "par":
                r[c]["nao_avaliados"] = sum(not l[c]["avaliado"] for l in linhas_real)
        r["kappa_validador_revisor"] = ab.kappa(
            [(l["validador"]["veredito"], l["revisor"]["veredito"]) for l in linhas_real])
        r["motivos_validador"] = dict(Counter(c for l in linhas_real if not l["validador"]["veredito"]
                                              for c in set(l["validador"]["problemas"])))
        r["motivos_revisor"] = dict(Counter(c for l in linhas_real if not l["revisor"]["veredito"]
                                            for c in set(l["revisor"]["problemas"]
                                                         + l["revisor"].get("criterios_falhos", []))))
        # D3: quantos reais o revisor (cego, thinking) rebaixaria para Fácil —
        # rótulo do banco x dificuldade_real, e quantos cumprem a regra (os dois
        # juízes true). Só existe nas rodadas com o revisor novo (r5+).
        pares = Counter(f"{l.get('dificuldade_rotulo')}->{l['revisor'].get('dificuldade_real')}"
                        for l in linhas_real if l["revisor"].get("dificuldade_real"))
        if pares:
            r["dificuldade_rotulo_x_real"] = dict(pares)
            r["rebaixaria_para_facil"] = sorted(
                l["id"] for l in linhas_real
                if l.get("dificuldade_rotulo") in ("Moderado", "Difícil")
                and l["revisor"].get("dificuldade_real") == "Fácil" and l["par"])
        out["real30"] = r
    if linhas_cal and linhas_real:
        aceita_ruim = bool(out["cal20"]["ruins_aceitas_pelo_par"])
        fr = out["real30"]["par"]["taxa_falsa_rejeicao"]
        out["precisa_ajuste"] = aceita_ruim or fr > LIMITE_FALSA_REJEICAO
        out["criterio"] = (f"ajustar se o par aceitar alguma ruim do cal20 ou rejeitar > "
                           f"{LIMITE_FALSA_REJEICAO:.0%} dos reais")
    return out


def salvar_prompts():
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PROMPTS_DIR / f"{aq.VERSAO_PROMPTS}.json"
    if not p.exists():
        p.write_text(json.dumps({k: getattr(aq, k) for k in (
            "GERADOR_ADDENDUM", "DIFICIL_ANOS_INICIAIS_ADDENDUM", "RESPOSTAS_USADAS_ADDENDUM",
            "VALIDADOR_SISTEMA", "VALIDADOR_USUARIO", "VALIDADOR_FASE2_SISTEMA", "VALIDADOR_FASE2",
            "REVISOR_SISTEMA", "REVISOR_USUARIO")}, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p.relative_to(ROOT))


def carregar_saida(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"teto_global": TETO_GLOBAL, "rodadas": {}}


def caminhos(args):
    if args.dry_run:
        d = Path(args.dir_dry_run)
        return {"saida": d / "calibracao_agentes.json", "cache": d / "cache.jsonl",
                "ledger": d / "ledger.json", "uso": d / "uso_api.jsonl", "ledger_gemini": d / "ledger_gemini.json",
                "uso_gemini": d / "uso_api_gemini.jsonl"}
    return {"saida": Path(args.saida), "cache": Path(args.cache), "ledger": Path(args.ledger),
            "uso": Path(args.log_uso), "ledger_gemini": LEDGER_GEMINI, "uso_gemini": ag.LOG_USO_GEMINI}


def executar(args):
    cam = caminhos(args)
    for k in ("saida", "cache", "ledger", "uso"):
        aq._garantir_gravavel(cam[k])
    if args.gemini_dificuldade:
        if not args.dry_run and args.max_chamadas_gemini is None:
            raise SystemExit("--max-chamadas-gemini é obrigatório fora do --dry-run.")
    elif not args.dry_run and args.max_chamadas is None and not args.metricas:
        raise SystemExit("--max-chamadas é obrigatório fora do --dry-run.")
    saida = carregar_saida(cam["saida"])
    rod = saida["rodadas"].setdefault(args.rodada, {"itens": {}})
    itens = []
    if args.conjunto in ("cal", "ambos", "todos", "regressao"):
        itens += carregar_cal20()
    if args.conjunto in ("real", "ambos", "todos"):
        reais, info_real = carregar_real()
        itens += reais
        saida["selecao_real"] = info_real | {"ids": [it["id"] for it in reais]}
    if args.conjunto in ("defeitos", "todos", "regressao"):
        itens += carregar_defeitos()
    if args.conjunto in ("d5", "todos", "regressao"):
        itens += carregar_d5()
    if args.conjunto == "dif":
        ref, info_ref = carregar_ref_dificuldade()
        itens += ref
        saida["selecao_ref_dificuldade"] = info_ref | {"ids": [it["id"] for it in ref]}
        # por rodada também: as métricas da rodada usam a seleção com que foram feitas
        rod["selecao_ref_dificuldade"] = saida["selecao_ref_dificuldade"]
    if args.ids:
        alvo = set(args.ids.split(","))
        itens = [it for it in itens if it["id"] in alvo]
    parada = None
    if args.gemini_dificuldade:
        gemini_dificuldade(args, saida, rod, cam)
        itens = []
    if args.reavaliar_revisor:
        base_rev = itens_da_rodada(saida, args.reavaliar_revisor)
        itens = [it for it in itens if it["id"] in base_rev]
    # Anota o filtro determinístico ATUAL nas linhas já julgadas (0 chamadas).
    for it in itens:
        if it["id"] in rod["itens"] and it["conjunto"] != "dif45":
            rod["itens"][it["id"]]["filtro_deterministico"] = filtro_deterministico(it["questao"], it.get("ano"),
                                                                                    it.get("habilidade"))
    if not args.metricas and not args.gemini_dificuldade:
        agentes = montar(args, args.rodada, cam)
        rod.setdefault("versoes_prompts", [])
        if aq.VERSAO_PROMPTS not in rod["versoes_prompts"]:
            rod["versoes_prompts"].append(aq.VERSAO_PROMPTS)
        rod["prompts"] = salvar_prompts() if not args.dry_run else None
        rod["modelos"] = agentes.modelos
        if args.nota:
            rod["nota"] = args.nota
        chamadas_ini = agentes.orcamento.chamadas
        for it in itens:
            # pior caso típico de um item: fase 1 + fase 2 + revisor. Na
            # reavaliação só do revisor (cache), o próprio Orcamento barra
            # qualquer envio que passe do teto (0 = só cache).
            if not agentes.orcamento.cabe(0 if args.reavaliar_revisor else 3):
                parada = f"orcamento: restam {agentes.orcamento.restante()} chamadas"
                break
            try:
                if args.reavaliar_revisor:
                    linha = revisar_de_novo(agentes, it, base_rev[it["id"]])
                else:
                    linha = julgar_item(agentes, it)
            except aq.OrcamentoEsgotado as exc:
                parada = f"orcamento_esgotado: {exc}"
                break
            linha["versao_prompts"] = aq.VERSAO_PROMPTS
            if it["conjunto"] != "dif45":
                linha["filtro_deterministico"] = filtro_deterministico(it["questao"], it.get("ano"),
                                                                       it.get("habilidade"))
            rod["itens"][it["id"]] = linha
            v, r = linha["validador"] or {"veredito": "-", "resposta": "-", "problemas": []}, linha["revisor"]
            print(f"  {it['id']:<22} {'boa ' if it['boa'] else 'RUIM'} val={str(v['veredito'])[0]}"
                  f"({v['resposta']}) rev={str(r['veredito'])[0]}({r['resposta']}) par={str(linha['par'])[0]}"
                  f" gab={linha['gabarito']} {sorted(set(v['problemas']))[:3]} {r.get('criterios_falhos', [])}"
                  f" dif={linha.get('dificuldade_rotulo')}->{r.get('dificuldade_real')}"
                  f"  [gasto={agentes.ledger.d['chamadas']}]", flush=True)
            # grava a cada item: uma queda não perde o que já foi pago
            cam["saida"].parent.mkdir(parents=True, exist_ok=True)
            cam["saida"].write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
        rod.setdefault("execucoes", []).append({
            "ts": aq.agora_iso(), "conjunto": args.conjunto, "ids": args.ids,
            "chamadas_reais": agentes.orcamento.chamadas - chamadas_ini, "cache_hits": agentes.hits,
            "prompt_tokens": agentes.orcamento.prompt_tokens, "completion_tokens": agentes.orcamento.completion_tokens,
            "tokens_estimados": agentes.orcamento.estimadas, "parada": parada or "concluido"})
        saida["gasto_total"] = agentes.ledger.d
    linhas = list(rod["itens"].values())
    if args.herdar:
        rod["herda_de"] = args.herdar
    if rod.get("herda_de"):
        # Reaproveita itens que esta rodada não refez (orçamento): só é
        # legítimo para itens cujo resultado a mudança de prompt não pode
        # alterar — quem roda decide e registra em --nota. Ficam marcados.
        origem = saida["rodadas"][rod["herda_de"]]["itens"]
        herdados = [dict(l, herdado_de=rod["herda_de"]) for i, l in origem.items() if i not in rod["itens"]]
        rod["herdados"] = sorted(l["id"] for l in herdados)
        linhas += herdados
    rod["metricas"] = metricas([l for l in linhas if l["conjunto"] == "cal20"],
                               [l for l in linhas if l["conjunto"] == "real30"])
    # dif45 só tem o revisor (validador None): fica fora das métricas do par.
    juizes = [l for l in linhas if l["conjunto"] != "dif45"]
    if any(l["conjunto"] in ("defeitos", "d5") for l in juizes) or args.conjunto == "todos":
        rod["metricas"]["r5"] = metricas_r5(juizes)
    if any(l["conjunto"] == "dif45" for l in linhas):
        rod["metricas"]["dificuldade"] = metricas_dificuldade(linhas, rod.get("selecao_ref_dificuldade"))
    saida["teto_global"] = args.teto_global
    cam["saida"].parent.mkdir(parents=True, exist_ok=True)
    cam["saida"].write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    imprimir(rod["metricas"], saida.get("gasto_total"))
    return saida


LEDGER_GEMINI = DIR_AGENTES / "calibracao_ledger_gemini.json"
TETO_GEMINI_ETAPA = 40


def alvo_gemini(linhas, n_facil=6):
    """Itens do dif45 que o Gemini julga: TODOS os M/D (os que a D3 poderia
    rebaixar — a regra 2 de 2 só importa neles) + n_facil F em rodízio por ano
    (controle: o Gemini também diz Fácil onde o banco diz Fácil?)."""
    dif = sorted((l for l in linhas if l["conjunto"] == "dif45"), key=lambda l: l["id"])
    md = [l for l in dif if l["dificuldade_rotulo"] in ("Moderado", "Difícil")]
    por_ano = {}
    for l in dif:
        if l["dificuldade_rotulo"] == "Fácil":
            por_ano.setdefault(l["ano"], []).append(l)
    fac = []
    while len(fac) < n_facil and any(por_ano.values()):
        for ano in sorted(por_ano):
            if por_ano[ano] and len(fac) < n_facil:
                fac.append(por_ano[ano].pop(0))
    return md + fac


def gemini_dificuldade(args, saida, rod, cam, arbitro=None):
    """Árbitro Gemini julga a dificuldade_real dos alvos (orçamento PRÓPRIO,
    ledger próprio, log de uso do Gemini). Grava em linha["gemini"]."""
    ref = {it["id"]: it for it in carregar_ref_dificuldade()[0]}
    linhas = [l for l in rod["itens"].values() if l["conjunto"] == "dif45"]
    ledger = Ledger(cam.get("ledger_gemini", LEDGER_GEMINI), args.teto_gemini)
    pedido = args.max_chamadas_gemini if args.max_chamadas_gemini is not None else (TETO_GEMINI_ETAPA if args.dry_run else 0)
    teto = min(pedido, ledger.restante())
    if arbitro is None:
        arbitro = ag.montar_arbitro(dry_run=args.dry_run, max_chamadas=teto,
                                    log_uso=cam.get("uso_gemini", ag.LOG_USO_GEMINI))
    alvo = [l for l in alvo_gemini(linhas, args.gemini_n_facil) if not (l.get("gemini") or {}).get("avaliado")]
    print(f"Gemini ({arbitro.modelo}): {len(alvo)} itens a julgar; teto {teto} (ledger {ledger.d['chamadas']}"
          f"/{ledger.teto})")
    parada = None
    for l in alvo:
        it = ref.get(l["id"])
        if it is None:
            continue
        if not arbitro.orcamento.cabe(1):
            parada = "orcamento"
            break
        antes = (arbitro.orcamento.chamadas, arbitro.orcamento.prompt_tokens, arbitro.orcamento.completion_tokens)
        try:
            r = arbitro.julgar_dificuldade(it["questao"], it["ano"], it["habilidade"], it["descricao"])
        except aq.OrcamentoEsgotado as exc:
            parada = f"orcamento_esgotado: {exc}"
            break
        finally:
            d = (arbitro.orcamento.chamadas - antes[0], arbitro.orcamento.prompt_tokens - antes[1],
                 arbitro.orcamento.completion_tokens - antes[2])
            if any(d) and not args.dry_run:
                ledger.somar(args.rodada, "arbitro_dificuldade", *d)
        l["gemini"] = r
        print(f"  {l['id']:<22} banco={l['dificuldade_rotulo']:<8} revisor={str(l['revisor'].get('dificuldade_real')):<8}"
              f" gemini={r.get('dificuldade_real')} etapas={r.get('etapas')}", flush=True)
        cam["saida"].write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    rod.setdefault("execucoes_gemini", []).append({
        "ts": aq.agora_iso(), "modelo": arbitro.modelo, "modelo_pedido": ag.MODELO_PEDIDO,
        "versao_prompt_dificuldade": ag.VERSAO_PROMPT_DIFICULDADE, "chamadas_reais": arbitro.orcamento.chamadas,
        "cache_hits": arbitro.hits, "prompt_tokens": arbitro.orcamento.prompt_tokens,
        "completion_tokens": arbitro.orcamento.completion_tokens, "parada": parada or "concluido"})
    if not args.dry_run:
        saida["gasto_total_gemini"] = ledger.d
    return parada


def imprimir(m, gasto):
    if "cal20" in m:
        c = m["cal20"]
        for j in ("validador", "revisor", "par"):
            x = c[j]
            print(f"cal20 {j:<9} VP={x['vp']} FN={x['fn']} FP={x['fp']} VN={x['vn']} "
                  f"(reprova ruins {x['reprova_ruins']}, aprova boas {x['aprova_boas']})")
        print(f"cal20 ruins aceitas pelo par: {c['ruins_aceitas_pelo_par']} | boas rejeitadas: "
              f"{c['boas_rejeitadas_pelo_par']} | kappa={c['kappa_validador_revisor']}")
    if "real30" in m:
        r = m["real30"]
        for j in ("validador", "revisor", "par"):
            print(f"real{r['n']} {j:<9} falsa rejeição {r[j]['rejeitados']}/{r['n']} = {r[j]['taxa_falsa_rejeicao']}"
                  f" IC95 {r[j]['ic95']}")
    if "r5" in m:
        r5 = m["r5"]
        for nome in ("cal20_rotulo_d5", "cal20_rotulo_log", "defeitos", "d5"):
            if nome in r5:
                for j in ("validador", "revisor", "par"):
                    x = r5[nome][j]
                    print(f"{nome:<16} {j:<9} reprova ruins {x['reprova_ruins']} aprova boas {x['aprova_boas']}"
                          f" | ruins aceitas {x['ruins_aceitas']} boas rejeitadas {x['boas_rejeitadas']}")
        if "real_falsa_rejeicao" in r5:
            x = r5["real_falsa_rejeicao"]
            print(f"reais: falsa rejeição bruta {x['bruta']}, ajustada {x['ajustada']} = {x['taxa_ajustada']}"
                  f" {x['rejeitados_ajustados']}")
        g = r5.get("geral") or {}
        if g:
            print(f"geral n={g['n']} kappa={g['kappa_validador_revisor']} reprova: val {g['reprova_validador']['taxa']}"
                  f" rev {g['reprova_revisor']['taxa']} par {g['reprova_par']['taxa']}")
        if r5.get("pipeline_filtro_mais_par"):
            x = r5["pipeline_filtro_mais_par"]
            print(f"pipeline (filtro exato + par) n={x['n']} VP={x['vp']} FN={x['fn']} FP={x['fp']} VN={x['vn']} "
                  f"| ruins aceitas {x['ruins_aceitas']} boas rejeitadas {x['boas_rejeitadas']}")
        print(f"metas r5: {'OK' if r5['metas_ok'] else 'FALHAS: ' + '; '.join(r5['metas_falhas'])}")
    if m.get("dificuldade"):
        d = m["dificuldade"]
        for quem in ("revisor", "gemini", "revisor_no_subconjunto_gemini"):
            if quem in d:
                x = d[quem]
                print(f"dificuldade {quem:<30} n={x['n']} exata={x['exata']} kappa_q={x['kappa_ponderado_quadratico']}"
                      f" kappa_l={x['kappa_ponderado_linear']} rebaixaria {x['rebaixaria_para_facil']['n']}/"
                      f"{x['rebaixaria_para_facil']['de']} = {x['rebaixaria_para_facil']['taxa']}")
                print(f"   matriz (banco -> julgado): {x['matriz_banco_x_julgado']}")
        if "regra_2_de_2" in d:
            print(f"regra 2 de 2: {d['regra_2_de_2']}")
        print(f"regra D3: {d['regra_d3']['regra']} ({d['regra_d3']['motivo']})")
    if gasto:
        print(f"gasto acumulado: {gasto['chamadas']} chamadas, {gasto['prompt_tokens']} tokens de entrada, "
              f"{gasto['completion_tokens']} de saída")


def construir_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rodada", default="r0")
    ap.add_argument("--conjunto", choices=("cal", "real", "ambos", "defeitos", "d5", "todos", "regressao", "dif"),
                    default="ambos",
                    help="ambos = cal + real; todos = cal + real + defeitos conhecidos + D5; regressao = cal + "
                         "defeitos + D5; dif = referência de dificuldade (45 reais, só o revisor)")
    ap.add_argument("--gemini-dificuldade", action="store_true",
                    help="árbitro Gemini julga a dificuldade dos itens dif45 da rodada (orçamento próprio)")
    ap.add_argument("--max-chamadas-gemini", type=int, default=None)
    ap.add_argument("--teto-gemini", type=int, default=TETO_GEMINI_ETAPA,
                    help="teto acumulado do ledger do Gemini na calibração")
    ap.add_argument("--gemini-n-facil", type=int, default=6)
    ap.add_argument("--ids", default=None, help="só estes ids (vírgula), ex.: R1-Q3,REAL-MT9084DH17MT")
    ap.add_argument("--nota", default=None, help="o que mudou nesta rodada")
    ap.add_argument("--reavaliar-revisor", default=None, metavar="RODADA",
                    help="refaz só o revisor (cache) sobre os itens da RODADA, mantendo o validador dela; "
                         "use com --max-chamadas 0 para garantir custo zero")
    ap.add_argument("--herdar", default=None,
                    help="rodada de onde vêm os itens não refeitos nesta (marcados como herdados)")
    ap.add_argument("--metricas", action="store_true", help="só recalcula as métricas da rodada (0 chamadas)")
    ap.add_argument("--listar-real", action="store_true", help="mostra a amostra real e sai (0 chamadas)")
    ap.add_argument("--teto-global", type=int, default=TETO_GLOBAL)
    ap.add_argument("--saida", default=str(SAIDA_PADRAO))
    ap.add_argument("--cache", default=str(CACHE_PADRAO))
    ap.add_argument("--ledger", default=str(LEDGER_PADRAO))
    ap.add_argument("--log-uso", default=str(USO_PADRAO))
    ap.add_argument("--dir-dry-run", default=str(DRYRUN_DIR))
    aq.adicionar_args_modelos(ap)
    return ap


def main(argv=None):
    args = construir_parser().parse_args(argv)
    if args.listar_real:
        reais, info = carregar_real()
        print(json.dumps(info, ensure_ascii=False))
        for it in reais:
            q = it["questao"]
            print(f"{it['id']:<22} {it['ano']} {it['habilidade']:<9} gab={q['resposta_correta']} "
                  f"{q['enunciado'][:110]!r} | {[q['alternativas'][L][:30] for L in 'ABCD']}")
        return 0
    executar(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
