"""Auditor da base de conhecimento: confiança de cada exemplo de data/train_curado.jsonl.

Por que auditar o que já está no treino: o relatório (src/relatorio_base.py)
mostrou que 40% das questões não têm conta que check_consistency consiga
conferir, e o erro conceitual do 9º H17 ("tem uma propriedade de X, logo é X")
não aparece como "inconsistente". Só um juiz que RESOLVE o item mede isso.

Para cada exemplo (todos ou --amostra N estratificada por habilidade):
  (a) filtros determinísticos PUROS (0 chamadas): estrutura, dependência
      visual, dados ausentes, check_consistency is False, verificador de
      geometria (se existir). Near-duplicata intra-base e contra val*.jsonl
      (contaminação) são só FLAGS — não baixam a confiança;
  (b) VALIDADOR (cego, alternativas permutadas, fase 2 só se bater) e
  (c) REVISOR — os dois SEMPRE (sem curto-circuito), porque o nível "media"
      precisa saber se exatamente um aprovou. --economico pula o revisor nos
      itens REAIS que o validador aprovou (decisão sua: muda o que "alta"
      significa para os reais);
  (d) confiança: "alta" (os dois true e filtro ok), "media" (exatamente um
      true), "baixa" (os dois false ou filtro reprovado), "nao_avaliado"
      (erro de API/parse — refeito na próxima rodada, nunca vira aprovação).

Saída: data/auditoria_base.jsonl (append-only, retomável; o registro mais
recente de cada idx vale; cache por hash do conteúdo + versão dos prompts:
rodar de novo custa zero). Resumo em outputs/agentes/auditoria_resumo.json,
estratificado por origem e por professor (os destilados do sabia-4-thinking
julgados pelo próprio sabia-4-thinking são o caso de maior risco de erro
correlacionado).

--montar produz data/train_curado_v3.jsonl SEM tocar no train_curado.jsonl:
  base auditada mais data/injecao_saeb.jsonl, com as regras:
  DECISÕES HUMANAS (H3, 2026-10-01) — Doc/decisoes_humanas.json (versionável)
     tem precedência sobre tudo abaixo: remover, manter ou manter com edições
     (resolução, dificuldade nos três lugares);
  ÁRBITRO (H4, 2026-10-01) — a remoção pela D2 só vale com o árbitro de outra
     família (Gemini, --arbitrar, orçamento próprio) também achando erro; sem
     arbitragem, ou com ele discordando, o item fica e vai para revisão humana;
  D2 (2026-10-01) — sai, de QUALQUER origem (inclusive real do banco), o
     exemplo com erro matemático CONFIRMADO: verificador aritmético exato
     contradiz o gabarito, OU validador cego reprova por motivo matemático E
     (revisor também reprova por motivo matemático OU filtro determinístico
     matemático reprova). Nunca sai quando o verificador exato CONFIRMA o
     gabarito (os juízes erram juntos: idx 881). Resolução vazia e ambiguidade
     de redação apontada pelo validador não são erro matemático. Lista com
     motivo e evidência: outputs/agentes/removidos_v3.jsonl;
  D3 (2026-10-01) — exemplo Moderado/Difícil com os dois juízes true e
     dificuldade_real=Fácil do revisor é REBAIXADO para Fácil nos três lugares
     (meta, difficulty, "Dificuldade: X." do prompt). Lista:
     outputs/agentes/rotulos_corrigidos_v3.jsonl. SUSPENSA desde o passo 2
     (REGRA_D3, --regra-d3): o revisor sem âncora rebaixaria 83% dos M/D
     reais do banco; com "revisor", item real nunca é rebaixado;
  sem confirmação — qualquer exemplo reprovado sem a confirmação da D2 FICA
     e vai para data/auditoria_revisao_humana.jsonl com a evidência (revisão
     do piloto 2: a regra anterior, que tirava destilados "baixa"/"media com
     problema matemático", removeu um item correto por um juiz só).
  (--remover-reais-baixa aplica a regra literal "confiança != baixa" também
  aos reais, exceto por resolução vazia.) Não auditado ou "nao_avaliado"
  fica (não há evidência contra).
  Injetadas — só entram as julgadas com os prompts atuais (na injeção ou em
     `injetar_questoes.py --rejulgar`); as outras vão para
     outputs/agentes/injetadas_excluidas_v3.jsonl.

--calibrar roda o par validador+revisor nas 20 questões do 9º H17 auditadas
por humano (APROVAR R1-Q1, R1-Q7, R2-Q1, R2-Q5, R2-Q9; REPROVAR as outras 15)
e mede o gate: o PAR reprova 15/15 ruins e aprova >= 4/5 boas, com IC de
Wilson 95%, kappa validador x revisor e P(ambos aprovam | ruim) contra o
produto das marginais (erro correlacionado). Saída: código 0 passa, 3 falha,
4 inconclusivo.

Uso:
    python src/auditar_base.py --dry-run --amostra 20
    python src/auditar_base.py --calibrar --max-chamadas 80
    python src/auditar_base.py --amostra 150 --max-chamadas 450
    python src/auditar_base.py --max-chamadas 4300            # base inteira
    python src/auditar_base.py --arbitrar --max-chamadas-arbitro 20   # Gemini, só os candidatos da D2
    python src/auditar_base.py --montar
    python src/auditar_base.py --montar --previa --removidos /tmp/x/rem.jsonl ...   # sem escrever a v3
    python src/auditar_base.py --resumo
"""

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import agentes_questoes as aq
import arbitro_gemini as ag
import diversidade
from schema_utils import DIFFICULTY_MAP, extract_questoes

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PADRAO = ROOT / "data" / "train_curado.jsonl"
VAL_PADRAO = [ROOT / "data" / n for n in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl")]
AUDITORIA_PADRAO = ROOT / "data" / "auditoria_base.jsonl"
INJECAO_PADRAO = ROOT / "data" / "injecao_saeb.jsonl"
V3_PADRAO = ROOT / "data" / "train_curado_v3.jsonl"
REVISAO_HUMANA_PADRAO = ROOT / "data" / "auditoria_revisao_humana.jsonl"
MONTAGEM_PADRAO = ROOT / "outputs" / "montagem_v3.json"
RESUMO_PADRAO = ROOT / "outputs" / "agentes" / "auditoria_resumo.json"
CALIBRACAO_PADRAO = ROOT / "outputs" / "agentes" / "calibracao.json"
DRYRUN_DIR = ROOT / "outputs" / "agentes" / "dryrun"
REMOVIDOS_PADRAO = ROOT / "outputs" / "agentes" / "removidos_v3.jsonl"
ROTULOS_PADRAO = ROOT / "outputs" / "agentes" / "rotulos_corrigidos_v3.jsonl"
RESOLUCAO_VAZIA_PADRAO = ROOT / "outputs" / "agentes" / "resolucao_vazia_v3.jsonl"
# H3 (2026-10-01): decisões HUMANAS, versionáveis (data/ está no .gitignore).
DECISOES_PADRAO = ROOT / "Doc" / "decisoes_humanas.json"
ARBITRAGEM_PADRAO = ag.ARBITRAGEM_D2

TESTES = ROOT / "outputs" / "testes_locais"
# Fonte de verdade humana (outputs/testes_locais/Log.txt).
CALIBRACAO = [(TESTES / "teste_20261001_103202.json", "R1", {1, 7}),
              (TESTES / "teste_20261001_103625.json", "R2", {1, 5, 9})]
# D5 (decisão do usuário, 2026-10-01): "só o gabarito pode ser verdadeiro".
# R2-Q5 ("todos os lados iguais. Como se chama?" com "Isósceles" entre as
# alternativas) foi aprovado no Log.txt, mas "Isósceles" é VERDADEIRO para o
# equilátero (definição inclusiva): é um distrator verdadeiro e agora deve ser
# reprovado. O rótulo humano original fica registrado em "rotulo_log".
# CONFIRMAR com o usuário; para voltar ao rótulo do Log.txt, esvazie o dict.
ROTULOS_REVISTOS_D5 = {"R2-Q5": False}

SEVERIDADE = {"baixa": 3, "nao_avaliado": 2, "media": 1, "alta": 0}
PROTEGIDAS = ("real", "sintetico")


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def carregar_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            try:
                out.append(json.loads(linha))
            except json.JSONDecodeError:  # linha final truncada por interrupção
                continue
    return out


def _assistant(ex):
    return next((m.get("content", "") for m in ex.get("messages", []) if m.get("role") == "assistant"), "")


def _user(ex):
    return next((m.get("content", "") for m in ex.get("messages", []) if m.get("role") == "user"), "")


def questoes_do_exemplo(ex):
    try:
        obj = json.loads(_assistant(ex))
    except (json.JSONDecodeError, TypeError):
        return None
    return [q for q in extract_questoes(obj) if isinstance(q, dict)]


def hash_exemplo(ex):
    import hashlib
    return hashlib.sha1(_assistant(ex).encode("utf-8")).hexdigest()


def origem(meta):
    codigo = str(meta.get("codigo_item") or "")
    if meta.get("destilado") or codigo.startswith(("DIST-", "INJ-")):
        return "destilado"
    if meta.get("sintetico") or codigo.startswith("SINT-"):
        return "sintetico"
    # O que não é destilado nem sintético é transcrição humana (INEP ou lote de
    # professor; o relatório mostrou 0 casos de origem desconhecida).
    return "real"


def descricao_do_prompt(ex):
    import re
    m = re.search(r"Habilidade: \S+ — (.*)\. Dificuldade:", _user(ex))
    return m.group(1) if m else ""


def amostra_estratificada(exemplos, n, seed=42):
    """n índices espalhados por (ano, habilidade): round-robin entre as
    habilidades, item sorteado dentro de cada uma. Garante que 150 itens
    cubram as 79 habilidades em vez de 9 delas.

    A ordem das habilidades INTERCALA os anos (1ª habilidade de cada ano,
    depois a 2ª de cada ano...). Por quê: 1º/3º/4º usam códigos BNCC largos
    (3, 3 e 1 habilidades, mas 428 exemplos) contra 22-26 habilidades SAEB do
    2º/5º/9º; com a ordem só sorteada, uma amostra menor que 79 (o piloto usou
    60) deixava o 4º ano inteiro de fora."""
    rng = random.Random(f"amostra|{seed}")
    grupos = defaultdict(list)
    for i, ex in enumerate(exemplos):
        m = ex.get("meta", {})
        grupos[(m.get("ano"), m.get("habilidade"))].append(i)
    por_ano = defaultdict(list)
    for k in sorted(grupos, key=lambda k: (str(k[0]), str(k[1]))):
        por_ano[str(k[0])].append(k)
    anos = sorted(por_ano)
    rng.shuffle(anos)
    for a in anos:
        rng.shuffle(por_ano[a])
    chaves = [por_ano[a][j] for j in range(max(len(v) for v in por_ano.values()))
              for a in anos if j < len(por_ano[a])]
    for k in chaves:
        rng.shuffle(grupos[k])
    escolhidos = []
    while len(escolhidos) < n and any(grupos[k] for k in chaves):
        for k in chaves:
            if grupos[k] and len(escolhidos) < n:
                escolhidos.append(grupos[k].pop())
    return sorted(escolhidos)


# ---------------------------------------------------------------------------
# Confiança
# ---------------------------------------------------------------------------

def nivel_confianca(filtro_reprovado, val, rev):
    """alta | media | baixa | nao_avaliado (ver docstring do módulo).

    rev None só acontece no modo econômico (real aprovado pelo validador)."""
    if filtro_reprovado:
        return "baixa"
    if val is None or not val.get("avaliado") or (rev is not None and not rev.get("avaliado")):
        return "nao_avaliado"
    v = bool(val["veredito"])
    if rev is None:
        return "alta" if v else "baixa"
    r = bool(rev["veredito"])
    if v and r:
        return "alta"
    if v != r:
        return "media"
    return "baixa"


def _resumo_juiz(j):
    if j is None:
        return None
    # "status" (V/F/I/G por alternativa, letras ORIGINAIS) entra no registro
    # desde a revisão do piloto 2: é a evidência de QUAIS alternativas cada
    # juiz julgou verdadeiras (o "detalhe" do LLM cita letras da permutação
    # dele, não as originais).
    return {k: j.get(k) for k in ("veredito", "avaliado", "resposta_calculada", "problemas", "confianca",
                                  "fases", "criterios_falhos", "erro", "sugestoes", "dificuldade_real",
                                  "status")
            if k in j}


def _codigos(*juizes):
    cods = []
    for j in juizes:
        for p in (j or {}).get("problemas") or []:
            if p.get("codigo"):
                cods.append(p["codigo"])
    return cods


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------

def caminhos(args):
    d = Path(args.dir_dry_run) if args.dry_run else None
    def esc(valor, padrao, nome):
        return Path(valor) if valor else (d / nome if d else padrao)
    return {"auditoria": esc(args.auditoria, AUDITORIA_PADRAO, "auditoria_base.jsonl"),
            "injecao": esc(args.injecao, INJECAO_PADRAO, "injecao_saeb.jsonl"),
            "v3": esc(args.saida_v3, V3_PADRAO, "train_curado_v3.jsonl"),
            "revisao_humana": esc(args.revisao_humana, REVISAO_HUMANA_PADRAO, "auditoria_revisao_humana.jsonl"),
            "montagem": esc(args.montagem, MONTAGEM_PADRAO, "montagem_v3.json"),
            "resumo": esc(args.resumo_json, RESUMO_PADRAO, "auditoria_resumo.json"),
            "calibracao": esc(args.calibracao_json, CALIBRACAO_PADRAO, "calibracao.json"),
            "uso": esc(args.log_uso, aq.LOG_USO_PADRAO, "uso_api.jsonl"),
            "removidos": esc(args.removidos, REMOVIDOS_PADRAO, "removidos_v3.jsonl"),
            "rotulos": esc(args.rotulos, ROTULOS_PADRAO, "rotulos_corrigidos_v3.jsonl"),
            "resolucao_vazia": esc(args.resolucao_vazia, RESOLUCAO_VAZIA_PADRAO, "resolucao_vazia_v3.jsonl"),
            "arbitragem": esc(getattr(args, "arbitragem", None), ARBITRAGEM_PADRAO, "arbitragem_d2.jsonl"),
            "uso_arbitro": esc(getattr(args, "log_uso_arbitro", None), ag.LOG_USO_GEMINI, "uso_api_gemini.jsonl"),
            # a decisão humana vale igual no dry-run (é entrada, não saída)
            "decisoes": Path(getattr(args, "decisoes_humanas", None) or DECISOES_PADRAO)}


def ultimos_registros(path):
    """Registro mais recente por idx (append-only: o último vence)."""
    reg = {}
    for r in carregar_jsonl(path):
        if "idx" in r:
            reg[r["idx"]] = r
    return reg


def _near_dup_intra(exemplos, alvo_idx):
    """{idx: [outros idx da mesma (ano, hab) que são near-duplicata]} só para alvo_idx."""
    por_chave = defaultdict(list)
    for i, ex in enumerate(exemplos):
        m = ex.get("meta", {})
        for q in questoes_do_exemplo(ex) or []:
            por_chave[(m.get("ano"), m.get("habilidade"))].append((i, q))
    out = {}
    for i in alvo_idx:
        m = exemplos[i].get("meta", {})
        qs = questoes_do_exemplo(exemplos[i]) or []
        viz = set()
        for j, outra in por_chave[(m.get("ano"), m.get("habilidade"))]:
            if j != i and any(diversidade.e_near_duplicata(q, outra) for q in qs):
                viz.add(j)
        out[i] = sorted(viz)
    return out


def auditar_exemplo(agentes, ex, idx, val_por_chave, near_dup, economico=False):
    """Registro de auditoria de UM exemplo (pode ter várias questões: o nível
    final é o PIOR entre elas)."""
    meta = ex.get("meta", {})
    ano, hab = meta.get("ano"), meta.get("habilidade")
    org = origem(meta)
    antes = (agentes.orcamento.chamadas, agentes.orcamento.prompt_tokens, agentes.orcamento.completion_tokens)
    qs = questoes_do_exemplo(ex)
    base = {"ts": aq.agora_iso(), "idx": idx, "codigo_item": meta.get("codigo_item"), "ano": ano,
            "habilidade": hab, "origem": org, "professor": meta.get("professor"),
            "hash_questao": hash_exemplo(ex), "versao_prompts": aq.VERSAO_PROMPTS,
            "versao_filtros": aq.VERSAO_FILTROS, "versao_juizes": aq.VERSAO_JUIZES,
            "rubrica_dificuldade": aq.RUBRICA_DIFICULDADE, "modelos": dict(agentes.modelos)}
    if not qs:
        return dict(base, filtros={"estrutura": False, "reprovado": ["json_invalido"]}, validador=None,
                    revisor=None, vereditos={"validador": None, "revisor": None}, confianca="baixa",
                    problemas=["json_invalido"], problemas_matematicos=[], problemas_pedagogicos=[],
                    revisao_humana=org in PROTEGIDAS, uso={"chamadas": 0, "prompt_tokens": 0,
                                                           "completion_tokens": 0})
    descricao = descricao_do_prompt(ex)
    por_q = []
    for q in qs:
        filtros = aq.filtros_auditoria(q, subtema=meta.get("subtema"))
        filtros["near_dup_base"] = near_dup.get(idx, [])
        filtros["contamina_val"] = any(diversidade.e_near_duplicata(q, v) for v in val_por_chave.get((ano, hab), []))
        # Os juízes rodam MESMO com filtro reprovado (antes eram pulados): a
        # regra D2 só remove um item com erro matemático CONFIRMADO pelo
        # validador cego, e o rótulo D3 precisa da dificuldade real do revisor.
        # O nível continua "baixa" quando um filtro reprova.
        val = agentes.validar(q)
        rev = None
        pular_rev = economico and org == "real" and val["veredito"] and not filtros["reprovado"]
        if not pular_rev:
            rev = agentes.revisar(q, ano, hab, descricao, meta.get("subtema"), meta.get("dificuldade"))
        nivel = nivel_confianca(filtros["reprovado"], val, rev)
        por_q.append({"filtros": filtros, "validador": _resumo_juiz(val), "revisor": _resumo_juiz(rev),
                      "confianca": nivel, "revisor_pulado_economico": bool(val and rev is None)})
    pior = max(por_q, key=lambda r: SEVERIDADE[r["confianca"]])
    cods = sorted(set(_codigos(*[r["validador"] for r in por_q], *[r["revisor"] for r in por_q])
                      + [f"filtro:{f}" for r in por_q for f in r["filtros"]["reprovado"]]))
    mat = [c for c in cods if c in aq.CODIGOS_MATEMATICOS or c.startswith("filtro:")]
    ped = [c for c in cods if c in aq.CODIGOS_PEDAGOGICOS]

    def _and(papel):
        vs = [r[papel]["veredito"] if r[papel] and r[papel].get("avaliado") else None for r in por_q]
        return None if any(v is None for v in vs) else all(vs)

    depois = (agentes.orcamento.chamadas, agentes.orcamento.prompt_tokens, agentes.orcamento.completion_tokens)
    reg = dict(base, filtros=pior["filtros"], validador=pior["validador"], revisor=pior["revisor"],
               vereditos={"validador": _and("validador"), "revisor": _and("revisor")},
               confianca=pior["confianca"], problemas=cods, problemas_matematicos=mat,
               problemas_pedagogicos=ped, dificuldade_rotulo=meta.get("dificuldade"),
               dificuldade_real=[(r["revisor"] or {}).get("dificuldade_real") for r in por_q],
               revisao_humana=org in PROTEGIDAS and pior["confianca"] in ("baixa", "media"),
               uso={"chamadas": depois[0] - antes[0], "prompt_tokens": depois[1] - antes[1],
                    "completion_tokens": depois[2] - antes[2]})
    if any(r["revisor_pulado_economico"] for r in por_q):
        reg["modo_economico"] = True
    if len(por_q) > 1:
        reg["por_questao"] = por_q
    return reg


def refiltrar(reg, ex):
    """Reaplica os filtros determinísticos ATUAIS a um registro em cache (0
    chamadas). O que os juízes disseram continua no registro; o nível de
    confiança é recalculado (mesma regra de nivel_confianca) quando o conjunto
    de filtros reprovados MUDA — nos dois sentidos:
      - filtro novo reprova (ex.: resolucao_supoe_dado): o nível cai;
      - filtro antigo deixa de reprovar (ABSOLVIÇÃO, H2 de 2026-10-01: com a
        regra das duas casas decimais o verificador aritmético passa a
        CONFIRMAR o idx 1348, "5/11 = 0,45"): o "filtro:..." sai dos problemas
        e o nível volta ao que os juízes deram. Antes só o primeiro sentido
        existia, e um item absolvido continuaria "baixa" para sempre.

    Por quê: o cache da auditoria é por (conteúdo, versão dos prompts). Sem
    isto, um filtro novo nunca alcançaria os exemplos já auditados e o rótulo
    errado ficaria na v3."""
    meta = ex.get("meta", {})
    qs = questoes_do_exemplo(ex) or []
    filtros_q = [aq.filtros_auditoria(q, subtema=meta.get("subtema")) for q in qs]
    velhos = reg.get("filtros") or {}
    antes = set(velhos.get("reprovado") or [])
    agora = set().union(*[set(f["reprovado"]) for f in filtros_q]) if filtros_q else set(antes)
    novos = sorted(agora - antes)
    absolvidos = sorted(antes - agora - {"estrutura", "json_invalido"})
    novo = dict(reg, ts=aq.agora_iso(), versao_filtros=aq.VERSAO_FILTROS,
                uso={"chamadas": 0, "prompt_tokens": 0, "completion_tokens": 0})
    if len(filtros_q) == 1:
        # o "confirma" do verificador aritmético também é evidência (impede a
        # D2 de remover um item correto por erro conjunto dos juízes)
        extra = {"aritmetica": filtros_q[0].get("aritmetica")}
        if filtros_q[0].get("geometria_d5"):
            extra["geometria_d5"] = filtros_q[0]["geometria_d5"]
        novo["filtros"] = dict(velhos, **extra)
    if not novos and not absolvidos:
        return novo
    manter = {"near_dup_base": velhos.get("near_dup_base", []), "contamina_val": velhos.get("contamina_val", False)}
    if reg.get("por_questao") and len(reg["por_questao"]) == len(filtros_q):
        por_q = []
        for f, r in zip(filtros_q, reg["por_questao"]):
            f2 = dict(f, **manter)
            por_q.append(dict(r, filtros=f2, confianca=nivel_confianca(f2["reprovado"], r.get("validador"),
                                                                         r.get("revisor"))))
        pior = max(por_q, key=lambda r: SEVERIDADE[r["confianca"]])
        novo["por_questao"] = por_q
        filtros, conf = pior["filtros"], pior["confianca"]
    else:
        filtros = dict(filtros_q[0] if len(filtros_q) == 1 else velhos, **manter)
        if len(filtros_q) != 1:
            filtros["reprovado"] = sorted(agora)
        conf = nivel_confianca(filtros["reprovado"], reg.get("validador"), reg.get("revisor"))
    tirar = {f"filtro:{k}" for k in absolvidos}
    por = {f"filtro:{k}" for k in novos}
    cods = sorted((set(reg.get("problemas") or []) - tirar) | por)
    mat = sorted((set(reg.get("problemas_matematicos") or []) - tirar) | por)
    return dict(novo, filtros=filtros, confianca=conf, problemas=cods, problemas_matematicos=mat,
                revisao_humana=origem(meta) in PROTEGIDAS and conf in ("baixa", "media"),
                refiltrado={"de": reg.get("confianca"), "filtros_novos": novos, "filtros_absolvidos": absolvidos})


def precisa_refazer(reg):
    """Registro que não vale como cache: "nao_avaliado", sem juízo do
    validador (erro de API/parse, ou registro antigo de quando um filtro
    reprovado pulava os juízes) ou de outra VERSAO_JUIZES. Sem o validador, a
    D2 não pode confirmar erro matemático. Estrutura/JSON inválido não tem o
    que julgar."""
    if reg.get("confianca") == "nao_avaliado":
        return True
    reprov = (reg.get("filtros") or {}).get("reprovado") or []
    if "estrutura" in reprov or "json_invalido" in reprov:
        return False
    v = reg.get("validador")
    if not (v and v.get("avaliado")):
        return True
    # A decisão calculada a partir das respostas mudou (ex.: G do validador
    # passou a reprovar): refaz. Com --cache-juizes a mensagem é a mesma e a
    # resposta vem do cache — custo zero.
    return reg.get("versao_juizes") != aq.VERSAO_JUIZES


def auditar(args, agentes=None):
    p = caminhos(args)
    for k in ("auditoria", "resumo", "uso"):
        aq._garantir_gravavel(p[k])
    if not args.dry_run and args.max_chamadas is None and agentes is None:
        raise SystemExit("--max-chamadas é obrigatório fora do --dry-run (as chamadas custam dinheiro).")
    exemplos = carregar_jsonl(args.train)
    feitos = ultimos_registros(p["auditoria"])
    if getattr(args, "idx", None):
        # Revisão do passo 2: auditar SÓ os itens apontados (ex.: os reais com
        # gabarito errado que a calibração achou — MT5023MH05MT, MT9036DH12TD —
        # e que nunca tinham passado pela auditoria), sem pagar a amostra toda.
        alvo = sorted({int(x) for x in str(args.idx).split(",") if x.strip()})
        fora = [i for i in alvo if not 0 <= i < len(exemplos)]
        if fora:
            raise SystemExit(f"--idx fora da base: {fora}")
    elif args.idx_registrados:
        # Reauditoria da MESMA amostra (ex.: depois de mudar os prompts): os idx
        # que já têm registro no arquivo, qualquer que seja a versão.
        alvo = sorted(i for i in feitos if isinstance(i, int) and 0 <= i < len(exemplos))
    elif args.amostra:
        alvo = amostra_estratificada(exemplos, args.amostra, args.seed)
    else:
        alvo = list(range(len(exemplos)))
    # Cache por (conteúdo, ano, habilidade): o revisor julga o alinhamento com a
    # habilidade (C1), então o mesmo texto em outra habilidade é outro juízo.
    por_hash = {(r["hash_questao"], r.get("ano"), r.get("habilidade")): r for r in feitos.values()
                if r.get("versao_prompts") == aq.VERSAO_PROMPTS and not precisa_refazer(r)}
    pendentes, a_refiltrar = [], []
    for i in alvo:
        r = feitos.get(i)
        if r and r.get("hash_questao") == hash_exemplo(exemplos[i]) and r.get("versao_prompts") == aq.VERSAO_PROMPTS \
                and not precisa_refazer(r):
            if r.get("versao_filtros") != aq.VERSAO_FILTROS:
                a_refiltrar.append(i)
            continue
        pendentes.append(i)
    val_por_chave = defaultdict(list)
    for vp in args.val:
        for ex in carregar_jsonl(vp):
            m = ex.get("meta", {})
            val_por_chave[(m.get("ano"), m.get("habilidade"))].extend(questoes_do_exemplo(ex) or [])
    near = _near_dup_intra(exemplos, pendentes)
    if agentes is None:
        agentes = aq.montar_agentes(args, simulado=args.dry_run, log_uso=p["uso"])
    if args.cache_juizes and getattr(agentes, "cache_juizes", None) is None:
        # Respostas já pagas dos juízes (ex.: calibração com os MESMOS prompts):
        # mensagem idêntica = 0 chamadas. Novas respostas vão para o 1º arquivo.
        agentes.cache_juizes = aq.CacheRespostas(args.cache_juizes)

    print(f"Auditoria ({'DRY-RUN, cliente simulado' if args.dry_run else 'API REAL'}) — alvo {len(alvo)} exemplos, "
          f"{len(alvo) - len(pendentes)} já auditados (cache), {len(pendentes)} pendentes; "
          f"orçamento {args.max_chamadas} chamadas; prompts {aq.VERSAO_PROMPTS}")
    print(f"  saída: {p['auditoria']}")
    parada = None
    novos = Counter()
    t_ini = time.monotonic()
    with aq.abrir_para_gravar(p["auditoria"], "a") as f:
        mudaram = Counter()
        for i in a_refiltrar:
            reg = refiltrar(feitos[i], exemplos[i])
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            if reg.get("refiltrado"):
                mudaram[f"{reg['refiltrado']['de']}->{reg['confianca']}"] += 1
        if a_refiltrar:
            print(f"  filtros {aq.VERSAO_FILTROS} reaplicados a {len(a_refiltrar)} registros em cache "
                  f"(0 chamadas); mudaram de nível: {dict(mudaram) or 0}")
        for n, i in enumerate(pendentes, 1):
            ex = exemplos[i]
            m = ex.get("meta", {})
            h = (hash_exemplo(ex), m.get("ano"), m.get("habilidade"))
            if h in por_hash:  # mesmo conteúdo em outro idx: reaproveita, 0 chamadas
                reg = dict(por_hash[h], idx=i, codigo_item=ex.get("meta", {}).get("codigo_item"),
                           ts=aq.agora_iso(), cache_de=por_hash[h]["idx"],
                           uso={"chamadas": 0, "prompt_tokens": 0, "completion_tokens": 0})
                if reg.get("versao_filtros") != aq.VERSAO_FILTROS:
                    reg = refiltrar(reg, ex)
            else:
                nq = len(questoes_do_exemplo(ex) or [None])
                # Pior caso: f1 + f2 + revisor por questão. Com cache dos juízes não
                # há pré-checagem: acerto não gasta, e uma resposta paga antes de o
                # teto parar o exemplo fica no cache (não se perde).
                if getattr(agentes, "cache_juizes", None) is None and not agentes.orcamento.cabe(3 * nq):
                    parada = "orcamento_insuficiente_para_exemplo"
                    break
                try:
                    reg = auditar_exemplo(agentes, ex, i, val_por_chave, near, args.economico)
                except aq.OrcamentoEsgotado as exc:
                    parada = f"orcamento_esgotado: {exc}"
                    break
                if not precisa_refazer(reg):
                    por_hash[h] = reg
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            f.flush()
            novos[reg["confianca"]] += 1
            if n % 10 == 0 or n == len(pendentes):
                print(f"  [{n}/{len(pendentes)}] {dict(novos)} | chamadas {agentes.orcamento.chamadas}")
    resumo = resumo_auditoria(p["auditoria"], agentes.orcamento.resumo(), parada, args)
    resumo["tempo_s_desta_execucao"] = round(time.monotonic() - t_ini, 1)
    if getattr(agentes, "cache_juizes", None) is not None:
        resumo["cache_juizes"] = agentes.cache_juizes.resumo()
    resumo["exemplos_novos_desta_execucao"] = sum(novos.values())
    p["resumo"].parent.mkdir(parents=True, exist_ok=True)
    p["resumo"].write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    _imprimir_resumo(resumo)
    return resumo


def resumo_auditoria(path, uso=None, parada=None, args=None):
    regs = list(ultimos_registros(path).values())
    por_conf = Counter(r["confianca"] for r in regs)
    estrat = defaultdict(Counter)
    for r in regs:
        estrat[f"origem={r.get('origem')}"][r["confianca"]] += 1
        estrat[f"professor={r.get('professor') or '-'}"][r["confianca"]] += 1
        estrat[f"ano={r.get('ano')}"][r["confianca"]] += 1
    cods = Counter(c for r in regs for c in r.get("problemas", []))
    concord = Counter()
    for r in regs:
        v, rv = r["vereditos"].get("validador"), r["vereditos"].get("revisor")
        if v is not None and rv is not None:
            concord[f"val={v}|rev={rv}"] += 1
    return {"gerado_em": aq.agora_iso(), "versao_prompts": aq.VERSAO_PROMPTS, "n_auditados": len(regs),
            "dry_run": bool(getattr(args, "dry_run", False)), "parada": parada or "concluido",
            "confianca": dict(por_conf), "estratos": {k: dict(v) for k, v in sorted(estrat.items())},
            "problemas_mais_comuns": dict(cods.most_common(25)), "concordancia_juizes": dict(concord),
            "revisao_humana": sum(1 for r in regs if r.get("revisao_humana")),
            "contamina_val": sum(1 for r in regs if (r.get("filtros") or {}).get("contamina_val")),
            "uso_desta_execucao": uso}


def _imprimir_resumo(r):
    print("\n=== Resumo da auditoria (todos os registros do arquivo) ===")
    print(f"Auditados: {r['n_auditados']} | confiança: {r['confianca']}")
    print(f"Revisão humana: {r['revisao_humana']} | contaminação de val: {r['contamina_val']}")
    print(f"Concordância validador x revisor: {r['concordancia_juizes']}")
    if r.get("uso_desta_execucao"):
        u = r["uso_desta_execucao"]
        print(f"Esta execução: {u['chamadas']} chamadas (teto {u['max_chamadas']}), "
              f"{u['prompt_tokens']} + {u['completion_tokens']} tokens")
    print(f"Parada: {r['parada']}")


# ---------------------------------------------------------------------------
# Montagem da v3
# ---------------------------------------------------------------------------

def _questoes_do_registro(reg):
    """[(filtros, validador, revisor)] de cada questão do registro (exemplo
    com várias questões guarda "por_questao"; senão o próprio registro)."""
    if reg.get("por_questao"):
        return [(q.get("filtros") or {}, q.get("validador"), q.get("revisor")) for q in reg["por_questao"]]
    return [(reg.get("filtros") or {}, reg.get("validador"), reg.get("revisor"))]


def _codigos_erro(juiz, resolucao_vazia, supoe_dado=False, papel="validador"):
    """Códigos de ERRO matemático de um juiz (D2).

    - Sem resposta única por falta de dados (figura perdida, dado ausente) é
      item INCOMPLETO, não errado — EXCETO quando o filtro determinístico viu
      a resolução do autor SUPOR o dado que falta ("como o hexágono é
      regular..."): aí o exemplo ensina a conta com um dado inventado
      (resolucao_usa_dado_ausente). Antes esse caso (idx 1368) só saía pela
      regra antiga "baixa", sem evidência registrada.
    - VALIDADOR que atribui a falta de resposta única à REDAÇÃO
      (pergunta_ambigua) não está apontando conta ou classe errada: vai para
      revisão humana (idx 599, MT9013: o "ou" do enunciado; gabarito 4 certo
      na leitura usual). Vale só para o validador, o juiz do rigor
      matemático; o revisor marca enunciado_ambiguo junto com erros reais
      (idx 561).
    - Com resolução vazia, os códigos sobre a resolução só repetem que ela
      não existe."""
    cods = {p.get("codigo") for p in (juiz or {}).get("problemas") or [] if isinstance(p, dict)}
    erro = cods & aq.CODIGOS_ERRO_MATEMATICO
    nao_unica = {"gabarito_sem_resposta_unica", "resposta_nao_unica"}
    if cods & aq.CODIGOS_INCOMPLETO:
        erro -= nao_unica
        if supoe_dado:
            erro.add("resolucao_usa_dado_ausente")
    if papel == "validador" and cods & aq.CODIGOS_AMBIGUIDADE:
        erro -= nao_unica
    if resolucao_vazia:
        erro -= aq.CODIGOS_RESOLUCAO
    return sorted(erro)


def _aritmetica(filtros):
    a = (filtros or {}).get("aritmetica")
    return a if isinstance(a, dict) else {}


def erro_matematico_confirmado(reg):
    """D2 (decisão de 2026-10-01), com o critério corrigido na revisão do
    piloto 2. (True, evidência) quando, em alguma questão do exemplo:

    (a) um VERIFICADOR ARITMÉTICO EXATO (aq.verificar_aritmetica) contradiz o
        gabarito — conta exata, não opinião: basta sozinho para CANDIDATAR o
        item (ex.: idx 1052, "307 por extenso", que os dois juízes novos
        aprovaram juntos). O idx 1348 (5/11 com gabarito "0,45") NÃO é mais
        caso de erro: pela H2 o verificador o CONFIRMA; ou
    (b) o VALIDADOR cego reprova por motivo matemático E (o REVISOR também
        reprova por motivo matemático OU um filtro determinístico matemático —
        consistência, geometria, resolução que supõe dado — reprova).

    E NUNCA quando o verificador exato CONFIRMA o gabarito: os dois juízes são
    da mesma família e erram juntos (idx 881, "terça + 10 dias": os dois
    contaram "sexta", o gabarito "quinta" está certo). A concordância deles
    não é independente; aí o item fica e vai para revisão humana.

    Vale para QUALQUER origem, inclusive itens reais (ex.: MT5023MH05MT, com o
    gabarito errado no próprio banco). Resolução vazia não é erro matemático;
    ambiguidade de redação apontada pelo validador também não.

    Candidatar não é remover: pela H4 o --montar só tira o item com o árbitro
    de outra família (Gemini) também achando erro (decidir_com_arbitro)."""
    if not reg:
        return False, None
    for filtros, val, rev in _questoes_do_registro(reg):
        reprov = set(filtros.get("reprovado") or [])
        arit = _aritmetica(filtros)
        if arit.get("status") == "confirma":
            continue
        if arit.get("status") == "contradiz":
            return True, {"verificador_aritmetico": arit.get("detalhe"),
                          "validador": _codigos_erro(val, False) if val else [],
                          "revisor": _codigos_erro(rev, False, papel="revisor") if rev else [],
                          "filtros": sorted(reprov & aq.FILTROS_MATEMATICOS)}
        vazia = "resolucao_vazia" in reprov
        supoe = "resolucao_supoe_dado" in reprov
        if not (val and val.get("avaliado") and val.get("veredito") is False):
            continue
        cv = _codigos_erro(val, vazia, supoe)
        if not cv:
            continue
        cr = (_codigos_erro(rev, vazia, supoe, papel="revisor")
              if rev and rev.get("avaliado") and rev.get("veredito") is False else [])
        cf = sorted(reprov & aq.FILTROS_MATEMATICOS)
        if cr or cf:
            return True, {"validador": cv, "revisor": cr, "filtros": cf}
    return False, None


def verificador_confirma_contra_juizes(reg):
    """O verificador exato confirma o gabarito de uma questão que algum juiz
    reprovou (erro do juiz, não do item): vai para revisão humana."""
    for filtros, val, rev in _questoes_do_registro(reg or {}):
        if _aritmetica(filtros).get("status") == "confirma" and any(
                j and j.get("avaliado") and j.get("veredito") is False for j in (val, rev)):
            return True
    return False


# REGRA D3 fixada no passo 2 da recalibração (2026-10-01, rodada r7 de
# outputs/calibracao_agentes.json). Critério fixado ANTES dos dados: o revisor
# sozinho só rebaixa se, nos reais M/D do banco, rebaixar <= 10% com kappa
# ponderado >= 0,60; senão exige o árbitro Gemini (2 de 2) com os mesmos 10%;
# senão a D3 fica SUSPENSA. Medido em 44 reais (15 F/15 M/15 D; o código do
# banco traz a dificuldade): o revisor sem âncora diz Fácil para 24 de 29 M/D
# (83%; kappa quadrático 0,09); com o Gemini confirmando, ainda 12 de 29 = 41%
# (ver o relatório, seção 11). Rodada r8 (revisão do passo 2: referência sem
# os 5 itens defeituosos, que o revisor tinha reprovado com razão): 24 de 30 =
# 80% (kappa 0,09) e 2 de 2 = 12 de 30 = 40% — a conclusão não muda (seção 12). Nenhuma das regras cumpre o limite: SUSPENSA. Além
# disso o rótulo do banco é RELATIVO (cada habilidade tem ~1/3 de F, M e D,
# por cota), e a rubrica é absoluta: rebaixar pela rubrica mudaria a
# convenção que o modelo aprende dos itens reais.
REGRAS_D3 = ("suspensa", "revisor")
REGRA_D3 = "suspensa"


def deve_rebaixar_para_facil(reg, meta, regra=None):
    """D3: True quando o exemplo está rotulado Moderado/Difícil, TODAS as suas
    questões foram aprovadas pelos dois juízes (matematicamente corretas),
    nenhum filtro matemático reprovou e o revisor (cego, com dificuldade_real)
    disse Fácil para todas. Registros antigos (sem dificuldade_real) nunca
    rebaixam. Só rebaixa para Fácil — nenhuma outra direção.

    P3 (2026-10-01): só vale a dificuldade_real julgada SEM a âncora da
    dificuldade pedida (registro com rubrica_dificuldade atual). Os 59
    registros do piloto 2 foram julgados com "Dificuldade pedida" na mensagem
    e o revisor rebaixou 47% dos Moderado/Difícil: esse sinal fica suspenso
    até a reauditoria com o revisor novo e a regra D3 fixada na recalibração.

    Passo 2 (2026-10-01): `regra` (padrão REGRA_D3 = "suspensa") — suspensa
    nunca rebaixa. Com "revisor" (opção explícita, --regra-d3), item REAL
    nunca é rebaixado automaticamente: o rótulo do banco é a referência
    contra a qual o revisor foi medido (e errou 83% dos M/D); só uma decisão
    humana (Doc/decisoes_humanas.json, ex.: MT9018MH04MT) troca o rótulo de
    um real."""
    regra = regra or REGRA_D3
    if regra not in REGRAS_D3:
        raise ValueError(f"regra D3 desconhecida: {regra}")
    if regra == "suspensa":
        return False
    if origem(meta) == "real":
        return False
    if not reg or meta.get("dificuldade") not in ("Moderado", "Difícil"):
        return False
    if reg.get("rubrica_dificuldade") != aq.RUBRICA_DIFICULDADE:
        return False
    qs = _questoes_do_registro(reg)
    for filtros, val, rev in qs:
        if not (val and val.get("avaliado") and val.get("veredito") is True):
            return False
        if not (rev and rev.get("avaliado") and rev.get("veredito") is True):
            return False
        if set(filtros.get("reprovado") or []) & aq.FILTROS_MATEMATICOS:
            return False
        if rev.get("dificuldade_real") != "Fácil":
            return False
    return bool(qs)


_DIFFICULTY_JSON = re.compile(r'("difficulty"\s*:\s*")(EASY|MEDIUM|HARD)(")')


def trocar_dificuldade(ex, para, fonte):
    """Cópia do exemplo com a dificuldade `para` nos TRÊS lugares:
    meta.dificuldade, "difficulty" de cada questão do assistente e
    "Dificuldade: X." do prompt de usuário. Nada mais muda (sistema,
    enunciado, alternativas, resolução, gabarito, codigo_item). Devolve
    (novo, None) ou (None, motivo) se algum dos três lugares não bate — aí o
    exemplo segue como está (rótulo inconsistente é pior que rótulo antigo).
    Usada pela D3 (só para Fácil) e pelas decisões humanas (H3)."""
    meta = ex.get("meta", {})
    de = meta.get("dificuldade")
    if de not in DIFFICULTY_MAP or para not in DIFFICULTY_MAP or de == para:
        return None, "dificuldade_invalida_ou_igual"
    iu = next((i for i, m in enumerate(ex.get("messages", [])) if m.get("role") == "user"), None)
    ia = next((i for i, m in enumerate(ex.get("messages", [])) if m.get("role") == "assistant"), None)
    if iu is None or ia is None:
        return None, "sem_user_ou_assistant"
    user = ex["messages"][iu]["content"]
    rx_user = re.compile(rf"Dificuldade: {re.escape(de)}\.")
    if len(rx_user.findall(user)) != 1:
        return None, "prompt_sem_dificuldade_unica"
    ass = ex["messages"][ia]["content"]
    if not _DIFFICULTY_JSON.search(ass):
        return None, "assistant_sem_difficulty"
    alvo = DIFFICULTY_MAP[para]
    ass_novo = _DIFFICULTY_JSON.sub(rf"\g<1>{alvo}\g<3>", ass)
    try:
        qs = extract_questoes(json.loads(ass_novo))
    except json.JSONDecodeError:
        return None, "assistant_json_invalido"
    if not qs or any(not isinstance(q, dict) or q.get("difficulty") != alvo for q in qs):
        return None, "difficulty_inconsistente"
    msgs = [dict(m) for m in ex["messages"]]
    msgs[iu]["content"] = rx_user.sub(f"Dificuldade: {para}.", user)
    msgs[ia]["content"] = ass_novo
    novo = dict(ex, messages=msgs, meta=dict(meta, dificuldade=para,
                                             rotulo_corrigido={"de": de, "para": para, "fonte": fonte}))
    return novo, None


def rebaixar_para_facil(ex):
    """D3: trocar_dificuldade para Fácil, só a partir de Moderado/Difícil."""
    if ex.get("meta", {}).get("dificuldade") not in ("Moderado", "Difícil"):
        return None, "so_rebaixa_moderado_ou_dificil"
    return trocar_dificuldade(ex, "Fácil", "auditoria:revisor_dificuldade_real")


# ---------------------------------------------------------------------------
# Decisões HUMANAS (H3, 2026-10-01)
# ---------------------------------------------------------------------------
# Arquivo versionável Doc/decisoes_humanas.json. Precedência sobre TODA regra
# automática (D2, D3, árbitro): o usuário revisou a lista de 12 itens e decidiu
# manter só o MT9018MH04MT (com resolução cadastrada e rótulo Fácil) e remover
# os outros 11; o idx 1348 fica por H2. Cada decisão é ligada ao item por
# codigo_item + idx + hash_questao: se o conteúdo mudou, ela NÃO é aplicada
# (vai para "decisoes_nao_aplicadas" na montagem) — decidir sobre um texto que
# o usuário não viu seria pior que não decidir.
ACOES_HUMANAS = ("remover", "manter", "manter_editar")
# "alternativas" (2026-10-01): restauração das frações que a planilha converteu
# em data (D6). Só troca o TEXTO de letras existentes, com pré-condição sobre o
# texto atual de cada letra trocada — nunca muda o gabarito nem o nº de letras.
CAMPOS_EDITAVEIS = ("resolucao_passo_a_passo", "dificuldade", "alternativas")


def carregar_decisoes(path):
    """{codigo_item: decisão}. Arquivo ausente = sem decisões. Decisão
    malformada levanta (erro de quem editou o arquivo, não do item)."""
    path = Path(path)
    if not path.exists():
        return {}
    dados = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for d in dados.get("decisoes") or []:
        faltam = [k for k in ("codigo_item", "acao", "data", "autor", "motivo") if not d.get(k)]
        if faltam or d["acao"] not in ACOES_HUMANAS:
            raise ValueError(f"decisão humana inválida ({d.get('codigo_item')}): faltam {faltam} ou ação "
                             f"{d.get('acao')!r}")
        if d["acao"] == "manter_editar" and (not d.get("edicoes")
                                             or set(d["edicoes"]) - set(CAMPOS_EDITAVEIS)):
            raise ValueError(f"edição humana inválida em {d['codigo_item']}: só {CAMPOS_EDITAVEIS}")
        if d["codigo_item"] in out:
            raise ValueError(f"decisão humana duplicada: {d['codigo_item']}")
        out[d["codigo_item"]] = d
    return out


def decisao_aplicavel(dec, idx, ex):
    """(True, None) ou (False, motivo) — a decisão foi tomada sobre ESTE item?"""
    if dec.get("idx") is not None and idx is not None and dec["idx"] != idx:
        return False, f"idx_diferente (decisão {dec['idx']}, item {idx})"
    if dec.get("hash_questao") and dec["hash_questao"] != hash_exemplo(ex):
        return False, "conteudo_mudou (hash_questao diferente)"
    return True, None


def aplicar_edicoes_humanas(ex, dec):
    """(novo, None) ou (None, motivo). Edições suportadas: resolução do item
    (só item de UMA questão; pré-condições conferidas) e dificuldade (os três
    lugares, via trocar_dificuldade). Registra meta.decisao_humana."""
    ed = dec.get("edicoes") or {}
    pre = dec.get("pre_condicoes") or {}
    meta = ex.get("meta", {})
    ia = next((i for i, m in enumerate(ex.get("messages", [])) if m.get("role") == "assistant"), None)
    if ia is None:
        return None, "sem_assistant"
    try:
        obj = json.loads(ex["messages"][ia]["content"])
    except json.JSONDecodeError:
        return None, "assistant_json_invalido"
    qs = extract_questoes(obj)
    if len(qs) != 1 or not isinstance(qs[0], dict):
        return None, "edicao_so_em_item_de_uma_questao"
    q = qs[0]
    for campo in ("resolucao_passo_a_passo", "resposta_correta"):
        if campo in pre and str(q.get(campo) or "") != pre[campo]:
            return None, f"pre_condicao_{campo}"
    if "dificuldade" in pre and meta.get("dificuldade") != pre["dificuldade"]:
        return None, "pre_condicao_dificuldade"
    if "alternativas" in ed:
        troca = ed["alternativas"]
        pre_alt = pre.get("alternativas") or {}
        alts = q.get("alternativas") or {}
        if not isinstance(troca, dict) or not troca or set(troca) - set(alts):
            return None, "alternativas_edicao_invalida"
        # sem pré-condição por letra não aplica: trocar texto que o humano não viu
        # seria decidir às cegas (mesma lógica do hash_questao).
        if set(troca) - set(pre_alt):
            return None, "alternativas_sem_pre_condicao"
        for L in troca:
            if str(alts.get(L)) != str(pre_alt[L]):
                return None, f"pre_condicao_alternativa_{L}"
        for L, v in troca.items():
            alts[L] = str(v)
    novo = json.loads(json.dumps(ex, ensure_ascii=False))
    if "alternativas" in ed:
        novo["messages"][ia]["content"] = json.dumps(obj, ensure_ascii=False)
    if "resolucao_passo_a_passo" in ed:
        texto = str(ed["resolucao_passo_a_passo"]).strip()
        if len(texto) < aq.MIN_RESOLUCAO:
            return None, "resolucao_editada_vazia"
        q["resolucao_passo_a_passo"] = texto
        novo["messages"][ia]["content"] = json.dumps(obj, ensure_ascii=False)
    if "dificuldade" in ed and ed["dificuldade"] != meta.get("dificuldade"):
        novo, falha = trocar_dificuldade(novo, ed["dificuldade"], "decisao_humana")
        if novo is None:
            return None, f"dificuldade:{falha}"
        novo["meta"]["rotulo_corrigido"].update(data=dec.get("data"), autor=dec.get("autor"))
    novo["meta"]["decisao_humana"] = {"acao": dec["acao"], "data": dec.get("data"), "autor": dec.get("autor"),
                                      "motivo": dec.get("motivo"), "campos_editados": sorted(ed)}
    return novo, None


# ---------------------------------------------------------------------------
# Árbitro de outra família antes da remoção pela D2 (H4, 2026-10-01)
# ---------------------------------------------------------------------------

def carregar_arbitragens(path):
    """{(idx, hash_questao): registro mais recente} do arquivo do árbitro."""
    out = {}
    for r in carregar_jsonl(path):
        if "idx" in r and r.get("hash_questao"):
            out[(r["idx"], r["hash_questao"])] = r
    return out


def arbitro_confirma_erro(reg, arb):
    """(True|False|None, códigos). None = sem arbitragem válida (algum
    resultado não avaliado). O árbitro CONFIRMA quando, em alguma questão, ele
    também acha erro matemático pelo MESMO critério aplicado ao validador
    (_codigos_erro: incompleto não é errado; ambiguidade de redação não é erro
    matemático; dado suposto pela resolução + dados insuficientes é erro)."""
    if not arb:
        return None, []
    resultados = arb.get("resultados") or []
    regs_q = _questoes_do_registro(reg or {})
    if not resultados or any(not r.get("avaliado") for r in resultados):
        return None, []
    cods = []
    for k, r in enumerate(resultados):
        filtros = regs_q[k][0] if k < len(regs_q) else (reg or {}).get("filtros") or {}
        reprov = set(filtros.get("reprovado") or [])
        cods += _codigos_erro(r, "resolucao_vazia" in reprov, "resolucao_supoe_dado" in reprov)
    return bool(cods), sorted(set(cods))


def decidir_com_arbitro(org, reg, arb, manter_media=False, remover_reais_baixa=False):
    """decidir() + H4: a remoção por erro matemático confirmado (D2) só vale
    com o árbitro de outra família CONCORDANDO. Ele discordando, ou ainda sem
    arbitragem, o item FICA e vai para revisão humana (os dois sabiá erraram
    juntos no 881 e no 1348)."""
    acao, motivo = decidir(org, reg, manter_media, remover_reais_baixa)
    if acao != "remover" or motivo != "erro_matematico_confirmado":
        return acao, motivo
    conf, _ = arbitro_confirma_erro(reg, arb)
    if conf is None:
        return "manter", "d2_aguardando_arbitro_revisao_humana"
    if conf:
        return "remover", "erro_matematico_confirmado_arbitro"
    if arbitro_ve_item_incompleto(arb):
        # Revisão do passo 2: no 1080 o Gemini TAMBÉM viu que não há resposta
        # única, mas atribuiu à falta de dado (dados_insuficientes). Pelo
        # critério da D2 isso é item incompleto, não errado — o item fica —,
        # mas chamar de "árbitro discorda" escondia que ele confirmou o defeito.
        return "manter", "d2_arbitro_ve_item_incompleto_revisao_humana"
    return "manter", "d2_arbitro_discorda_revisao_humana"


def arbitro_ve_item_incompleto(arb):
    """O árbitro apontou resposta não única JUNTO com falta de dado (código de
    CODIGOS_INCOMPLETO) em alguma questão."""
    nao_unica = {"gabarito_sem_resposta_unica", "resposta_nao_unica"}
    for r in (arb or {}).get("resultados") or []:
        cods = {p.get("codigo") for p in r.get("problemas") or [] if isinstance(p, dict)}
        if cods & nao_unica and cods & aq.CODIGOS_INCOMPLETO:
            return True
    return False


def decidir(org, reg, manter_media=False, remover_reais_baixa=False):
    """('manter'|'remover', motivo).

    Só sai da v3 o erro matemático CONFIRMADO (D2), de qualquer origem.
    Revisão do piloto 2: a regra anterior — destilado "baixa", ou "media" com
    qualquer código matemático, sai — removia sem a confirmação que a D2
    exige e sem evidência registrada; tirou da prévia o DIST-H01-Difícil-00408
    (idx 1187, "dez mil quatrocentos e oito", correto) por um único juiz. Agora
    todo exemplo reprovado sem confirmação FICA e vai para a lista de revisão
    humana, com a evidência dos juízes; `manter_media` não tem mais efeito
    (mantido só para a CLI). `remover_reais_baixa` continua sendo a opção
    explícita do usuário para itens reais (exceto resolução vazia)."""
    if reg is None:
        return "manter", "nao_auditado"
    confirmado, _ = erro_matematico_confirmado(reg)
    if confirmado:
        return "remover", "erro_matematico_confirmado"
    c = reg.get("confianca")
    if verificador_confirma_contra_juizes(reg):
        return "manter", "verificador_confirma_gabarito_revisao_humana"
    if c == "alta":
        return "manter", "alta"
    if c == "nao_avaliado":
        return "manter", "nao_avaliado"
    if org in PROTEGIDAS:
        # D2: resolução vazia não é erro matemático. Se é só ela (os códigos dos
        # juízes, tirados os que falam da resolução, não acusam mais nada), nem
        # --remover-reais-baixa remove.
        reprov = set((reg.get("filtros") or {}).get("reprovado") or [])
        outros = set(reg.get("problemas") or []) - aq.CODIGOS_RESOLUCAO - aq.CODIGOS_INFORMATIVOS \
            - {"filtro:resolucao_vazia", "incoerencia_interna"}
        so_resolucao_vazia = reprov == {"resolucao_vazia"} and not outros
        if c == "baixa" and remover_reais_baixa and org == "real" and not so_resolucao_vazia:
            return "remover", "real_baixa_por_opcao"
        return "manter", f"{c}_protegido_revisao_humana"
    return "manter", f"{c}_sem_confirmacao_revisao_humana"


def injetada_valida(ex, qs=None):
    """(True, None) ou (False, motivo) para uma aceita de data/injecao_saeb.jsonl.

    Revisão do piloto 2: o --montar copiava TODAS as aceitas, e 6 das 10 do
    piloto 1 (prompts 7570dee3dd46, revisor antigo que via o gabarito) violam
    a D5 — INJ-9-H17-F-00001 é literalmente o "Acutângulo" que o usuário deu
    de exemplo. Critério geral: só entra a aceita julgada com os prompts
    ATUAIS, na injeção ou num rejulgamento (injetar_questoes --rejulgar, que
    grava meta.rejulgamento), e que o verificador aritmético exato não
    contradiz."""
    m = ex.get("meta", {})
    rj = m.get("rejulgamento") or {}
    if aq.VERSAO_PROMPTS not in (m.get("versao_prompts"), rj.get("versao_prompts")):
        return False, "julgada_com_prompts_antigos_sem_rejulgamento"
    for q in qs if qs is not None else (questoes_do_exemplo(ex) or []):
        if aq.verificar_aritmetica(q)["status"] == "contradiz":
            return False, "verificador_aritmetico"
        # P1 (2026-10-01): a mesma D5 determinística que barra o candidato na
        # injeção (distrator verdadeiro por superclasse ou por outro eixo).
        if aq.veredito_d5_geometria(q)["status"] == "reprovada":
            return False, "geometria_d5"
        if aq.alternativas_corrompidas_planilha(q):
            return False, "alternativa_corrompida_planilha"
    return True, None


def corrompido_planilha(ex):
    """D6 (revisão do passo 2, 2026-10-01): evidência {questão: {letra:
    {atual, proposta}}} quando alguma alternativa do exemplo é uma data que a
    planilha pôs no lugar de uma fração (aq.alternativas_corrompidas_planilha);
    None se não há. Ex.: MT90123MH25MT (idx 259), gabarito A =
    "2025-06-03 00:00:00", proposta "3/6"."""
    out = {}
    for k, q in enumerate(questoes_do_exemplo(ex) or []):
        c = aq.alternativas_corrompidas_planilha(q)
        if c:
            out[k] = {"alternativas": {L: {"atual": q["alternativas"][L], "proposta": f} for L, f in c.items()},
                      "gabarito": q.get("resposta_correta")}
    return out or None


def _registro_valido(regs, i, ex):
    """Registro de auditoria do idx `i` se for do MESMO conteúdo; com filtros de
    outra versão, refiltrado na hora (0 chamadas, não gravado)."""
    reg = regs.get(i)
    if reg is None or reg.get("hash_questao") != hash_exemplo(ex):
        return None, False  # auditoria de outro conteúdo (base mudou): não vale
    if reg.get("versao_filtros") != aq.VERSAO_FILTROS:
        return refiltrar(reg, ex), True
    return reg, False


def montar(args):
    """Monta a v3 SEM tocar no train_curado. Grava, além da v3:
    removidos_v3.jsonl (lista com o motivo e a evidência de cada removido),
    rotulos_corrigidos_v3.jsonl (D3 e decisões humanas), resolucao_vazia_v3.jsonl
    (itens com resolução vazia: NÃO removidos por isso, listados para completar
    a resolução).

    Ordem de precedência (2026-10-01):
      1. decisão HUMANA (Doc/decisoes_humanas.json, H3) — remover, manter ou
         manter com edições; nada automático se aplica ao item;
      2. D6: alternativa que é data da planilha (fração convertida pelo Excel)
         sai sem juiz nem árbitro, com a proposta de restauração na evidência;
      3. D2 com ÁRBITRO (H4): erro matemático confirmado só sai se o árbitro de
         outra família (Gemini, `--arbitrar`) também achar erro; sem arbitragem
         ou com ele discordando, fica e vai para revisão humana;
      4. D3 e o resto das regras automáticas.
    Registros de auditoria com filtros de outra versão são refiltrados na hora
    (0 chamadas; ex.: H2 absolve o 1348).

    --previa: as mesmas decisões, mas a v3 NÃO é escrita; a lista de revisão
    humana e o resumo vão para outputs/agentes/*_previa.* (ou para os caminhos
    dados nas opções)."""
    p = caminhos(args)
    previa = bool(getattr(args, "previa", False))
    if previa:
        p = dict(p, v3=None)
        if not args.montagem:
            p["montagem"] = p["removidos"].with_name("montagem_v3_previa.json")
        if not args.revisao_humana:
            p["revisao_humana"] = p["removidos"].with_name("revisao_humana_previa.jsonl")
    for k in ("v3", "revisao_humana", "montagem", "removidos", "rotulos", "resolucao_vazia"):
        if p[k] is None:
            continue
        aq._garantir_gravavel(p[k])
    base = carregar_jsonl(args.train)
    regs = ultimos_registros(p["auditoria"])
    decisoes = carregar_decisoes(p["decisoes"])
    arbitragens = carregar_arbitragens(p["arbitragem"])
    regra_d3 = getattr(args, "regra_d3", None) or REGRA_D3
    mantidos, removidos, humanos, rotulos, vazias = [], [], [], [], []
    dec_aplicadas, dec_nao_aplicadas = [], []
    d2 = Counter()
    n_refiltrados = 0
    motivos = defaultdict(Counter)
    import distill_teacher as dt
    for i, ex in enumerate(base):
        meta = ex.get("meta", {})
        org = origem(meta)
        reg, refiltrado = _registro_valido(regs, i, ex)
        n_refiltrados += int(refiltrado)
        qs = questoes_do_exemplo(ex) or []
        vazia = any(aq.defeitos_resolucao(q) == ["resolucao_vazia"] for q in qs)
        info = {"idx": i, "codigo_item": meta.get("codigo_item"), "ano": meta.get("ano"),
                "habilidade": meta.get("habilidade"), "origem": org}
        dec = decisoes.get(meta.get("codigo_item"))
        if dec is not None:
            ok, porque = decisao_aplicavel(dec, i, ex)
            novo = None
            if ok and dec["acao"] == "manter_editar":
                novo, porque = aplicar_edicoes_humanas(ex, dec)
                ok = novo is not None
            if not ok:
                dec_nao_aplicadas.append(dict(info, acao=dec["acao"], motivo=porque))
            else:
                evid = {k: dec.get(k) for k in ("acao", "data", "autor", "motivo", "fonte")}
                dec_aplicadas.append(dict(info, acao=dec["acao"]))
                motivos[org][f"{'remover' if dec['acao'] == 'remover' else 'manter'}:decisao_humana_"
                             f"{dec['acao']}"] += 1
                if dec["acao"] == "remover":
                    removidos.append(dict(info, professor=meta.get("professor"),
                                          confianca=(reg or {}).get("confianca"), motivo="decisao_humana",
                                          evidencia=evid, problemas=(reg or {}).get("problemas", []),
                                          versao_prompts=(reg or {}).get("versao_prompts")))
                    if vazia:
                        vazias.append(dict(info, acao="remover", nota="removido por decisão humana"))
                    continue
                if novo is not None:
                    rc = novo["meta"].get("rotulo_corrigido")
                    if rc and rc.get("fonte") == "decisao_humana":
                        rotulos.append(dict(info, fonte="decisao_humana", de=rc["de"], para=rc["para"],
                                            data=dec.get("data"), autor=dec.get("autor"),
                                            nota=dec.get("motivo")))
                    ex = novo
                elif vazia:
                    vazias.append(dict(info, acao="manter", nota="mantido por decisão humana, resolução vazia"))
                mantidos.append(ex)
                continue
        # D6 (revisão do passo 2): alternativa que é data da planilha. Sai da
        # v3 sem juiz nem árbitro — não é opinião sobre a matemática, é o
        # gabarito gravado como "2025-06-03 00:00:00"; a questão não tem
        # resposta (H1: questão errada sai). A restauração dia/mês ("3/6") vai
        # como PROPOSTA na evidência: restaurar é decisão humana (Doc/
        # decisoes_humanas.json, que tem precedência e já foi aplicada acima).
        planilha = corrompido_planilha(ex)
        if planilha is not None:
            ev = planilha[0] if len(planilha) == 1 else planilha
            motivos[org]["remover:alternativa_corrompida_planilha"] += 1
            removidos.append(dict(info, professor=meta.get("professor"), confianca=(reg or {}).get("confianca"),
                                  motivo="alternativa_corrompida_planilha", evidencia=ev,
                                  problemas=(reg or {}).get("problemas", []),
                                  versao_prompts=(reg or {}).get("versao_prompts")))
            if vazia:
                vazias.append(dict(info, acao="remover", nota="removido pela D6 (data da planilha)"))
            continue
        arb = arbitragens.get((i, hash_exemplo(ex)))
        acao, motivo = decidir_com_arbitro(org, reg, arb, args.manter_media, args.remover_reais_baixa)
        if motivo.startswith(("erro_matematico_confirmado", "d2_")):
            d2[motivo] += 1
        motivos[org][f"{acao}:{motivo}"] += 1
        if vazia:
            vazias.append(dict(info, acao=acao,
                               nota="resolução vazia não é erro matemático (D2): não removido por isso"))
        if acao == "manter":
            if deve_rebaixar_para_facil(reg, meta, regra_d3):
                novo, falha = rebaixar_para_facil(ex)
                rot = dict(info, fonte="base", de=meta.get("dificuldade"), para="Fácil",
                           dificuldade_real=reg.get("dificuldade_real"), versao_prompts=reg.get("versao_prompts"))
                if novo is not None:
                    ex = novo
                    rotulos.append(rot)
                else:
                    rotulos.append(dict(rot, para=meta.get("dificuldade"), nao_aplicado=falha))
            mantidos.append(ex)
        else:
            conf, evid = erro_matematico_confirmado(reg)
            if evid is not None and arb is not None:
                evid = dict(evid, arbitro={"modelo": arb.get("modelo"),
                                           "codigos": arbitro_confirma_erro(reg, arb)[1],
                                           "respostas": [r.get("resposta_calculada")
                                                         for r in arb.get("resultados") or []]})
            removidos.append(dict(info, professor=meta.get("professor"), confianca=reg.get("confianca"),
                                  motivo=motivo, evidencia=evid, problemas=reg.get("problemas", []),
                                  versao_prompts=reg.get("versao_prompts")))
        if reg is not None and (reg.get("revisao_humana") or
                                (acao == "manter" and motivo.endswith("revisao_humana"))):
            # Revisão do piloto 2: destilados reprovados SEM confirmação da D2 não
            # saem mais; ficam na v3 e entram nesta lista, com a evidência.
            humanos.append(dict(reg, motivo_montagem=motivo, arbitragem=arb))
    vistos = set()
    for ex in mantidos:
        for q in questoes_do_exemplo(ex) or []:
            vistos.add(dt.normalizar(q.get("enunciado", "")))
    codigos = {ex.get("meta", {}).get("codigo_item") for ex in mantidos}
    injetados, inj_dup, inj_excl = [], 0, []
    pendente = Path(str(p["injecao"]) + ".pendente")
    if pendente.exists():
        print(f"AVISO: {pendente.name} existe (injeção interrompida): rode a injeção de novo para recuperar a "
              "aceita do diário antes de montar; ela NÃO entra nesta montagem.")
    for ex in carregar_jsonl(p["injecao"]):
        qs = questoes_do_exemplo(ex) or []
        chave = dt.normalizar(qs[0].get("enunciado", "")) if qs else ""
        cod = ex.get("meta", {}).get("codigo_item")
        if not qs or chave in vistos or cod in codigos:
            inj_dup += 1
            continue
        dec = decisoes.get(cod)
        if dec is not None and dec["acao"] == "remover" and decisao_aplicavel(dec, None, ex)[0]:
            m = ex.get("meta", {})
            dec_aplicadas.append({"idx": None, "codigo_item": cod, "acao": "remover"})
            inj_excl.append({"codigo_item": cod, "ano": m.get("ano"), "habilidade": m.get("habilidade"),
                             "motivo": "decisao_humana", "versao_prompts": m.get("versao_prompts")})
            continue
        ok, mot = injetada_valida(ex, qs)
        if not ok:
            m = ex.get("meta", {})
            inj_excl.append({"codigo_item": cod, "ano": m.get("ano"), "habilidade": m.get("habilidade"),
                             "motivo": mot, "versao_prompts": m.get("versao_prompts"),
                             "rejulgamento": m.get("rejulgamento")})
            continue
        vistos.add(chave)
        codigos.add(cod)
        injetados.append(ex)
        rc = ex.get("meta", {}).get("rotulo_corrigido")
        if rc:
            m = ex["meta"]
            rotulos.append({"idx": None, "codigo_item": cod, "ano": m.get("ano"), "habilidade": m.get("habilidade"),
                            "origem": "injecao_saeb", "fonte": "injecao", "de": rc.get("de"), "para": rc.get("para"),
                            "fonte_rotulo": rc.get("fonte"), "nota": rc.get("nota"),
                            "dificuldade_real": [m.get("dificuldade_real") or (m.get("rejulgamento") or {})
                                                 .get("dificuldade_real")],
                            "versao_prompts": rc.get("versao_prompts")})
    if not previa:
        with aq.abrir_para_gravar(p["v3"], "w") as f:
            for ex in mantidos + injetados:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    p["injetadas_excluidas"] = p["removidos"].with_name("injetadas_excluidas_v3.jsonl")
    for chave, linhas in (("revisao_humana", humanos), ("removidos", removidos), ("rotulos", rotulos),
                          ("resolucao_vazia", vazias), ("injetadas_excluidas", inj_excl)):
        with aq.abrir_para_gravar(p[chave], "w") as f:
            for r in linhas:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    aplicados = [r for r in rotulos if "nao_aplicado" not in r]
    rel = {"gerado_em": aq.agora_iso(), "base": str(args.train), "auditoria": str(p["auditoria"]),
           "injecao": str(p["injecao"]), "saida": None if previa else str(p["v3"]), "previa": previa,
           "decisoes_humanas": str(p["decisoes"]), "arbitragem": str(p["arbitragem"]),
           "regras": {"manter_media": args.manter_media, "remover_reais_baixa": args.remover_reais_baixa,
                      "precedencia": "decisão humana > D6 (data da planilha) > D2 com árbitro > D3/automáticas",
                      "D2": "erro matemático confirmado (verificador aritmético exato contradiz, OU validador cego "
                            "E [revisor OU filtro matemático]) só sai com o ÁRBITRO de outra família (Gemini) "
                            "também achando erro; sem arbitragem ou com ele discordando, fica e vai para revisão "
                            "humana; nunca sai quando o verificador exato confirma o gabarito; resolução vazia e "
                            "ambiguidade de redação apontada pelo validador não contam",
                      "H2": "duas casas decimais (arredondado ou truncado) valem para dízima/irracional",
                      "D6": "alternativa que é data da planilha (fração convertida pelo Excel) sai da v3 sem "
                            "juiz nem árbitro; a fração dia/mês vai como proposta de restauração (decisão humana)",
                      "injetadas": "só entram as julgadas com os prompts atuais (na injeção ou em "
                                   "rejulgamento), que o verificador aritmético não contradiz e sem distrator "
                                   "verdadeiro pela D5 determinística",
                      "D3": (f"regra {regra_d3}: " + (
                          "nenhum rebaixamento automático (passo 2: nem o revisor sem âncora nem o 2 de 2 com o "
                          "Gemini cumprem o limite de 10% de rebaixamento falso nos reais M/D do banco)"
                          if regra_d3 == "suspensa" else
                          "rebaixa para Fácil (meta, difficulty, prompt) itens NÃO reais com os dois juízes true e "
                          "dificuldade_real=Fácil do revisor (rubrica sem âncora)"))},
           "n_base": len(base), "n_auditados_validos": sum(1 for i, ex in enumerate(base)
                                                          if i in regs and regs[i].get("hash_questao") == hash_exemplo(ex)),
           "n_registros_refiltrados_na_montagem": n_refiltrados,
           "n_decisoes_humanas_aplicadas": len(dec_aplicadas), "decisoes_humanas_aplicadas": dec_aplicadas,
           "n_decisoes_humanas_nao_aplicadas": len(dec_nao_aplicadas), "decisoes_nao_aplicadas": dec_nao_aplicadas,
           "d2": dict(d2),
           "n_mantidos_base": len(mantidos), "n_removidos": len(removidos),
           "n_removidos_erro_matematico": sum(r["motivo"].startswith("erro_matematico_confirmado")
                                              for r in removidos),
           "n_removidos_decisao_humana": sum(r["motivo"] == "decisao_humana" for r in removidos),
           "n_removidos_planilha": sum(r["motivo"] == "alternativa_corrompida_planilha" for r in removidos),
           "n_rotulos_corrigidos": len(aplicados), "n_rotulos_nao_aplicados": len(rotulos) - len(aplicados),
           "n_resolucao_vazia_listados": len(vazias),
           "n_injetados": len(injetados),
           "n_injetados_descartados_duplicata": inj_dup, "n_injetados_excluidos": len(inj_excl),
           "injetados_excluidos": inj_excl, "n_total_v3": len(mantidos) + len(injetados),
           "n_revisao_humana": len(humanos), "motivos_por_origem": {k: dict(v) for k, v in motivos.items()},
           "arquivos": {k: str(p[k]) for k in ("removidos", "rotulos", "resolucao_vazia", "revisao_humana",
                                               "injetadas_excluidas")},
           "removidos": removidos}
    p["montagem"].parent.mkdir(parents=True, exist_ok=True)
    p["montagem"].write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'PRÉVIA (v3 NÃO escrita)' if previa else 'v3 montada em ' + str(p['v3'])}: {len(mantidos)} da base "
          f"(de {len(base)}; {len(removidos)} removidos: {rel['n_removidos_decisao_humana']} por decisão humana, "
          f"{rel['n_removidos_planilha']} pela D6 (data da planilha), "
          f"{rel['n_removidos_erro_matematico']} por erro matemático confirmado com árbitro) + {len(injetados)} "
          f"injetados = {len(mantidos) + len(injetados)}")
    print(f"D2: {dict(d2) or 0} | decisões humanas aplicadas: {len(dec_aplicadas)}, não aplicadas: "
          f"{len(dec_nao_aplicadas)} | registros refiltrados: {n_refiltrados}")
    print(f"Rótulos corrigidos: {len(aplicados)} | resolução vazia listada: {len(vazias)} | "
          f"revisão humana: {len(humanos)} | injetadas excluídas: {len(inj_excl)}")
    print(f"Listas: {p['removidos']} | {p['rotulos']} | {p['resolucao_vazia']}")
    print(f"{args.train} NÃO foi alterado.")
    return rel


def candidatos_d2(args, p=None):
    """[(idx, exemplo, registro)] que a D2 removeria e que ainda não têm
    arbitragem válida (mesmo conteúdo). Decisões humanas aplicáveis tiram o
    item da lista (o humano já decidiu)."""
    p = p or caminhos(args)
    base = carregar_jsonl(args.train)
    regs = ultimos_registros(p["auditoria"])
    decisoes = carregar_decisoes(p["decisoes"])
    arbitragens = carregar_arbitragens(p["arbitragem"])
    out = []
    for i, ex in enumerate(base):
        dec = decisoes.get(ex.get("meta", {}).get("codigo_item"))
        if dec is not None and decisao_aplicavel(dec, i, ex)[0]:
            continue
        if corrompido_planilha(ex) is not None:
            continue  # a D6 já tira o item sem árbitro: não gasta chamada
        reg, _ = _registro_valido(regs, i, ex)
        if decidir(origem(ex.get("meta", {})), reg, args.manter_media, args.remover_reais_baixa) != \
                ("remover", "erro_matematico_confirmado"):
            continue
        arb = arbitragens.get((i, hash_exemplo(ex)))
        if arbitro_confirma_erro(reg, arb)[0] is None:
            out.append((i, ex, reg))
    return out


def arbitrar(args, arbitro=None):
    """H4: o árbitro de outra família resolve às cegas cada item que a D2
    removeria. Grava um registro por item em outputs/agentes/arbitragem_d2.jsonl
    (append-only; o mais recente por idx+conteúdo vale), consumido pelo
    --montar. Orçamento PRÓPRIO (--max-chamadas-arbitro), separado do da
    Maritaca."""
    p = caminhos(args)
    aq._garantir_gravavel(p["arbitragem"])
    if not args.dry_run and args.max_chamadas_arbitro is None and arbitro is None:
        raise SystemExit("--max-chamadas-arbitro é obrigatório fora do --dry-run (as chamadas custam dinheiro).")
    alvo = candidatos_d2(args, p)
    print(f"Árbitro ({'DRY-RUN' if args.dry_run else 'API REAL'}): {len(alvo)} itens da D2 sem arbitragem; "
          f"orçamento {args.max_chamadas_arbitro} chamadas")
    if not alvo:
        return {"arbitrados": 0, "parada": "nada_a_arbitrar"}
    if arbitro is None:
        arbitro = ag.montar_arbitro(dry_run=args.dry_run, max_chamadas=(40 if args.max_chamadas_arbitro is None else args.max_chamadas_arbitro),
                                    log_uso=p["uso_arbitro"], modelo=args.modelo_arbitro)
    feitos, parada = Counter(), None
    with aq.abrir_para_gravar(p["arbitragem"], "a") as f:
        for i, ex, reg in alvo:
            qs = questoes_do_exemplo(ex) or []
            if not arbitro.orcamento.cabe(len(qs)):
                parada = "orcamento_insuficiente_para_item"
                break
            try:
                resultados = [arbitro.resolver_cego(q) for q in qs]
            except aq.OrcamentoEsgotado as exc:
                parada = f"orcamento_esgotado: {exc}"
                break
            r = {"ts": aq.agora_iso(), "idx": i, "codigo_item": ex.get("meta", {}).get("codigo_item"),
                 "hash_questao": hash_exemplo(ex), "modelo": arbitro.modelo, "modelo_pedido": ag.MODELO_PEDIDO,
                 "versao_prompt_arbitro": ag.VERSAO_PROMPT_ARBITRO, "resultados": resultados,
                 "d2_evidencia": erro_matematico_confirmado(reg)[1]}
            conf, cods = arbitro_confirma_erro(reg, r)
            r["arbitro_confirma_erro"], r["codigos_arbitro"] = conf, cods
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            feitos["confirma" if conf else ("discorda" if conf is False else "nao_avaliado")] += 1
    res = {"arbitrados": sum(feitos.values()), "resultado": dict(feitos), "parada": parada or "concluido",
           "uso": arbitro.orcamento.resumo(), "modelo": arbitro.modelo}
    print(json.dumps(res, ensure_ascii=False))
    return res


# ---------------------------------------------------------------------------
# Calibração (gate antes de gastar na escala)
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    ph = k / n
    den = 1 + z * z / n
    centro = (ph + z * z / (2 * n)) / den
    meia = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centro - meia), 3), round(min(1.0, centro + meia), 3))


def kappa(pares):
    """Kappa de Cohen entre dois juízes binários: [(a, b), ...]."""
    n = len(pares)
    if n == 0:
        return None
    po = sum(a == b for a, b in pares) / n
    pa = sum(a for a, _ in pares) / n
    pb = sum(b for _, b in pares) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def carregar_calibracao(fontes=CALIBRACAO):
    itens = []
    for path, rotulo, boas in fontes:
        dados = json.loads(Path(path).read_text(encoding="utf-8"))
        n = 0
        for res in dados["resultados"]:
            for q in res["questoes"]:
                n += 1
                id_ = f"{rotulo}-Q{n}"
                itens.append({"id": id_, "ano": res["ano"], "habilidade": res["habilidade"],
                              "descricao": res.get("descricao", ""), "dificuldade": res.get("dificuldade"),
                              "questao": q, "boa": ROTULOS_REVISTOS_D5.get(id_, n in boas),
                              "rotulo_log": n in boas})
    return itens


def calibrar(args, agentes=None, itens=None):
    p = caminhos(args)
    aq._garantir_gravavel(p["calibracao"])
    if not args.dry_run and args.max_chamadas is None and agentes is None:
        raise SystemExit("--max-chamadas é obrigatório fora do --dry-run.")
    itens = itens if itens is not None else carregar_calibracao()
    if agentes is None:
        agentes = aq.montar_agentes(args, simulado=args.dry_run, log_uso=p["uso"])
    linhas, parada = [], None
    for it in itens:
        if not agentes.orcamento.cabe(3):
            parada = "orcamento_insuficiente"
            break
        try:
            j = agentes.julgar(it["questao"], it["ano"], it["habilidade"], it["descricao"], None,
                               it["dificuldade"], curto_circuito=False)
        except aq.OrcamentoEsgotado as exc:
            parada = f"orcamento_esgotado: {exc}"
            break
        val, rev = j["validador"], j["revisor"]
        linhas.append({"id": it["id"], "boa": it["boa"], "validador": val["veredito"], "revisor": rev["veredito"],
                       "avaliado": bool(val.get("avaliado") and rev.get("avaliado")), "par": j["ambos"],
                       "problemas_validador": _codigos(val), "problemas_revisor": _codigos(rev),
                       "resposta_cega": val.get("resposta_calculada"), "gabarito": it["questao"].get("resposta_correta")})
        print(f"  {it['id']} ({'boa' if it['boa'] else 'ruim'}): val={val['veredito']} rev={rev['veredito']} "
              f"par={j['ambos']} {_codigos(val)[:2]}")
    ruins = [l for l in linhas if not l["boa"]]
    boas = [l for l in linhas if l["boa"]]
    tn = sum(not l["par"] for l in ruins)
    tp = sum(l["par"] for l in boas)
    nao_av = [l["id"] for l in linhas if not l["avaliado"]]

    def taxa(lst, campo):
        return round(sum(l[campo] for l in lst) / len(lst), 3) if lst else None

    pv, pr = taxa(ruins, "validador"), taxa(ruins, "revisor")
    completo = len(linhas) == len(itens) and not nao_av
    passa = completo and tn == len(ruins) and tp >= math.ceil(0.8 * len(boas))
    res = {"gerado_em": aq.agora_iso(), "dry_run": bool(args.dry_run), "versao_prompts": aq.VERSAO_PROMPTS,
           "modelos": agentes.modelos, "n": len(linhas), "n_esperado": len(itens), "parada": parada or "concluido",
           "par": {"reprova_ruins": f"{tn}/{len(ruins)}", "ic95_reprova_ruins": wilson(tn, len(ruins)),
                   "aprova_boas": f"{tp}/{len(boas)}", "ic95_aprova_boas": wilson(tp, len(boas))},
           "validador": {"aprova_ruins": pv, "aprova_boas": taxa(boas, "validador")},
           "revisor": {"aprova_ruins": pr, "aprova_boas": taxa(boas, "revisor")},
           "erro_correlacionado": {"p_ambos_aprovam_dado_ruim": taxa(ruins, "par"),
                                   "produto_marginais": round(pv * pr, 3) if pv is not None and pr is not None else None},
           "kappa_validador_revisor": kappa([(l["validador"], l["revisor"]) for l in linhas if l["avaliado"]]),
           "nao_avaliados": nao_av,
           "gate": "passa" if passa else ("inconclusivo" if not completo else "falha"),
           "criterio": (f"par reprova {len(ruins)}/{len(ruins)} ruins e aprova >= {math.ceil(0.8 * len(boas))}/"
                        f"{len(boas)} boas; todos avaliados (rótulos revistos pela D5: "
                        f"{sorted(ROTULOS_REVISTOS_D5)})"),
           "aviso": ("n pequeno: 15/15 tem IC95 inferior de ~0,80; os few-shots do validador seguem os PADRÕES "
                     "destas 20 questões, então o resultado aqui é otimista — confirme em mutantes e no "
                     "conjunto cego do rotulador independente."),
           "uso": agentes.orcamento.resumo(), "itens": linhas}
    p["calibracao"].parent.mkdir(parents=True, exist_ok=True)
    p["calibracao"].write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCalibração: par reprova ruins {res['par']['reprova_ruins']} (IC95 {res['par']['ic95_reprova_ruins']}), "
          f"aprova boas {res['par']['aprova_boas']} (IC95 {res['par']['ic95_aprova_boas']}) -> GATE {res['gate'].upper()}")
    print(f"kappa val x rev: {res['kappa_validador_revisor']} | P(ambos aprovam|ruim)="
          f"{res['erro_correlacionado']['p_ambos_aprovam_dado_ruim']} vs produto {res['erro_correlacionado']['produto_marginais']}")
    print(f"Detalhes: {p['calibracao']}")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def construir_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--montar", action="store_true", help="monta data/train_curado_v3.jsonl e sai (0 chamadas)")
    modo.add_argument("--calibrar", action="store_true", help="gate nas 20 questões auditadas do 9º H17")
    modo.add_argument("--resumo", action="store_true", help="só imprime o resumo do arquivo de auditoria")
    modo.add_argument("--arbitrar", action="store_true",
                      help="H4: o árbitro Gemini resolve às cegas os itens que a D2 removeria (orçamento próprio)")
    ap.add_argument("--train", default=str(TRAIN_PADRAO), help="base lida (somente leitura)")
    ap.add_argument("--val", nargs="*", default=[str(v) for v in VAL_PADRAO])
    ap.add_argument("--amostra", type=int, default=None, help="N exemplos estratificados por habilidade")
    ap.add_argument("--idx", default=None,
                    help="audita só estes idx da base (vírgula), ex.: 139,401")
    ap.add_argument("--idx-registrados", action="store_true",
                    help="reaudita exatamente os idx que já têm registro no arquivo de auditoria (mesma amostra)")
    ap.add_argument("--cache-juizes", nargs="*", default=None,
                    help="arquivos .jsonl de respostas já pagas dos juízes (ex.: outputs/agentes/"
                         "calibracao_cache.jsonl); mensagem idêntica não chama a API; respostas novas vão "
                         "para o 1º arquivo")
    ap.add_argument("--previa", action="store_true",
                    help="com --montar: grava só as listas (removidos, rótulos, resolução vazia) e o resumo "
                         "da montagem, SEM escrever a v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--economico", action="store_true",
                    help="pula o revisor nos itens REAIS aprovados pelo validador (~-18%% de chamadas)")
    ap.add_argument("--manter-media", action="store_true",
                    help="SEM EFEITO desde a revisão do piloto 2 (só sai erro matemático confirmado, D2); "
                         "mantido para compatibilidade da CLI")
    ap.add_argument("--remover-reais-baixa", action="store_true",
                    help="na montagem, remove também itens reais com confiança baixa (padrão: revisão humana)")
    for nome in ("auditoria", "injecao", "saida-v3", "revisao-humana", "montagem", "resumo-json",
                 "calibracao-json", "log-uso", "removidos", "rotulos", "resolucao-vazia", "arbitragem",
                 "log-uso-arbitro"):
        ap.add_argument(f"--{nome}", default=None)
    ap.add_argument("--decisoes-humanas", default=str(DECISOES_PADRAO),
                    help="decisões humanas versionáveis (precedência sobre as automáticas)")
    ap.add_argument("--regra-d3", choices=REGRAS_D3, default=REGRA_D3,
                    help="D3 (rebaixar para Fácil): 'suspensa' (padrão, fixada no passo 2 pelos números da r7) ou "
                         "'revisor' (revisor sem âncora diz Fácil; nunca em item real)")
    ap.add_argument("--max-chamadas-arbitro", type=int, default=None,
                    help="teto DURO de chamadas ao árbitro Gemini (separado do da Maritaca)")
    ap.add_argument("--modelo-arbitro", default=ag.MODELO_ARBITRO,
                    help=f"padrão {ag.MODELO_ARBITRO} (pedido: {ag.MODELO_PEDIDO}, indisponível na API)")
    ap.add_argument("--dir-dry-run", default=str(DRYRUN_DIR))
    aq.adicionar_args_modelos(ap)
    return ap


def main(argv=None):
    args = construir_parser().parse_args(argv)
    if args.montar:
        montar(args)
        return 0
    if args.arbitrar:
        arbitrar(args)
        return 0
    if args.resumo:
        _imprimir_resumo(resumo_auditoria(caminhos(args)["auditoria"], args=args))
        return 0
    if args.calibrar:
        res = calibrar(args)
        return {"passa": 0, "falha": 3, "inconclusivo": 4}[res["gate"]]
    auditar(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
