"""Injeção de questões padrão SAEB na base de conhecimento, com veredito DUPLO.

A partir da LACUNA medida por src/relatorio_base.py (outputs/relatorio_base.json:
falta por dificuldade e por subtema de cada (ano, habilidade)), para cada
habilidade com déficit:

  1. planeja slots com diversidade.planejar_lote (somente leitura): subtemas
     menos cobertos primeiro, raciocínio/contexto girando, dificuldades que
     faltam; a letra do gabarito é sorteada para equilibrar A-E;
  2. GERADOR (1 chamada) -> filtros determinísticos (0 chamadas: estrutura,
     figura/dados ausentes, tamanho, consistência, duplicata, subtema,
     dependência visual, difficulty, near-duplicata contra base/aceitas,
     contaminação da validação, verificador de geometria se existir)
     -> VALIDADOR (cego; fase 2 só se a resposta cega bate)
     -> REVISOR (só se o validador aprovou)
  3. a questão ENTRA SÓ se os dois vereditos forem true. Não se corrige letra
     nem resolução: reprovou, gera de novo (até --tentativas-por-slot).
     Exceção de RÓTULO (D3, 2026-10-01): pedida Moderado/Difícil, aprovada
     pelos dois e julgada Fácil pelo revisor -> entra como Fácil (meta,
     difficulty e prompt de treino), ocupando um slot Fácil pendente; o slot
     Moderado/Difícil continua aberto. SUSPENSA desde o passo 2 da
     recalibração (--regra-d3, padrão auditar_base.REGRA_D3 = "suspensa"):
     o revisor sem âncora disse Fácil para 83% dos M/D reais do banco; a
     aprovada entra com a dificuldade PEDIDA e a dificuldade_real fica só
     registrada na meta.

Decisões de 2026-10-01 aplicadas aqui: Difícil também no 1º e 3º ano (D4);
duplicata SEMÂNTICA = mesmos números no enunciado + mesma resposta, contra
toda a base, a validação e as aceitas (T1); resposta não numérica repetida
demais na habilidade é barrada e o professor é avisado (T4); linha truncada
na saída vai para quarentena e a aceita é recuperada do diário .pendente
(T3); empate de dificuldade sorteado (T6).

Saídas (arquivos NOVOS; train_curado/train/val nunca são escritos):
  data/injecao_saeb.jsonl         aceitas, MESMO formato do treino. O prompt de
                                  usuário é USER_TEMPLATE + sufixo de diversidade
                                  (igual ao distill_teacher/inferência); meta com
                                  origem='injecao_saeb', vereditos, confiança,
                                  modelos, data, versao_prompts.
  outputs/injecao_rejeitadas.jsonl  rejeitadas com etapa e motivo.
  data/injecao_saeb_truncadas.jsonl fragmentos de linha truncada (quarentena, T3).
  outputs/agentes/uso_api.jsonl   uma linha por chamada (tokens, latência).
  outputs/agentes/injecao_resumo.json  taxas por etapa, custo por aceita.

Retomável/idempotente: os arquivos são append-only; ao rodar de novo, o que
já foi aceito conta para a meta e as tentativas anteriores contam para o teto
de tentativas da habilidade — nada é refeito.

Orçamento: --max-chamadas é teto DURO (obrigatório fora do --dry-run). Um
candidato só começa se couberem as 4 chamadas do pior caso (gerador +
validador f1 + f2 + revisor), para não pagar uma geração que não dá para julgar.

Uso:
    python src/injetar_questoes.py --dry-run --anos 2º,5º --meta-por-habilidade 3
    python src/injetar_questoes.py --anos 2º,5º --meta-por-habilidade 5 --max-chamadas 300
    python src/injetar_questoes.py --habilidades 9º:H17 --meta-por-habilidade 10 --max-chamadas 120
"""

import argparse
import contextlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import agentes_questoes as aq
import distill_teacher as dt
import diversidade
from extract_data import SYSTEM_PROMPT
from schema_utils import DIFFICULTY_MAP, extract_questao

ROOT = Path(__file__).resolve().parent.parent
RELATORIO_PADRAO = ROOT / "outputs" / "relatorio_base.json"
TRAIN_PADRAO = ROOT / "data" / "train_curado.jsonl"
VAL_PADRAO = [ROOT / "data" / n for n in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl")]
SAIDA_PADRAO = ROOT / "data" / "injecao_saeb.jsonl"
REJEITADAS_PADRAO = ROOT / "outputs" / "injecao_rejeitadas.jsonl"
RESUMO_PADRAO = ROOT / "outputs" / "agentes" / "injecao_resumo.json"
DRYRUN_DIR = ROOT / "outputs" / "agentes" / "dryrun"

DIFICULDADES = ("Fácil", "Moderado", "Difícil")
ORDEM_ANOS = ("1º", "2º", "3º", "4º", "5º", "9º")
SIGLA_DIF = {"Fácil": "F", "Moderado": "M", "Difícil": "D"}
CHAMADAS_PIOR_CASO = 4  # gerador + validador f1 + f2 + revisor
# D4 (decisão do usuário, 2026-10-01): Difícil também no 1º e no 3º ano.
ANOS_DIFICIL_LIBERADO = ("1º", "3º")
# T4: fração máxima das aceitas de uma habilidade com a MESMA resposta não
# numérica (classe/nome). Piso de 2 para metas pequenas.
FRACAO_MAX_MESMA_RESPOSTA = 0.25


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def carregar_jsonl(path):
    """Tolerante a linha final truncada (arquivo append-only interrompido)."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            try:
                out.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return out


def norm_ano(s):
    s = str(s).strip()
    return s if s.endswith("º") else re.sub(r"\D", "", s) + "º"


def parse_habilidades(texto):
    """'H17,9º:H17,9 H07' -> {(None,'H17'), ('9º','H17'), ('9º','H07')}."""
    out = set()
    for parte in (texto or "").split(","):
        parte = parte.strip()
        if not parte:
            continue
        m = re.match(r"^(\d+)\s*º?\s*[:| ]\s*(\S+)$", parte)
        out.add((norm_ano(m.group(1)), m.group(2).upper()) if m else (None, parte.upper()))
    return out


def descricoes_treino(exemplos):
    """Descrição mais frequente no PROMPT de treino de cada (ano, habilidade).

    Usa a do prompt (e não a do relatório) para que os exemplos injetados
    repitam exatamente a string que o modelo já vê no treino/inferência."""
    cont = defaultdict(Counter)
    for ex in exemplos:
        m = ex.get("meta", {})
        user = next((x.get("content", "") for x in ex.get("messages", []) if x.get("role") == "user"), "")
        mm = re.search(r"Habilidade: \S+ — (.*)\. Dificuldade:", user)
        if mm:
            cont[(m.get("ano"), m.get("habilidade"))][mm.group(1)] += 1
    return {k: c.most_common(1)[0][0] for k, c in cont.items()}


# ---------------------------------------------------------------------------
# Lacuna -> plano
# ---------------------------------------------------------------------------

def lacunas(relatorio, anos=None, habilidades=None, meta_por_hab=None, incluir_dif_ausente=False,
            liberar_dificil=ANOS_DIFICIL_LIBERADO):
    """Habilidades com déficit, das mais carentes (fração da meta faltando) para
    as menos. Devolve (lista, ignoradas).

    Por padrão só pede dificuldades que o BANCO REAL usa naquela habilidade,
    com uma exceção decidida pelo usuário em 2026-10-01 (D4): Difícil é pedida
    também no 1º e no 3º ano (`liberar_dificil`), onde o banco nunca usa — o
    gerador recebe instrução de dificuldade genuína dentro do ano e o revisor
    confere (C2). Outras dificuldades ausentes no banco (ex.: Fácil/Moderado no
    2º H21) continuam fora; --incluir-dificuldade-ausente libera todas."""
    lista, ignoradas = [], []
    for h in relatorio.get("habilidades", []):
        ano, hab = h["ano"], h["habilidade"]
        if anos and ano not in anos:
            continue
        if habilidades and (ano, hab) not in habilidades and (None, hab) not in habilidades:
            continue
        graus = [g for g in (h.get("banco") or {}).get("graus", []) if g in DIFICULDADES]
        if graus and ano in (liberar_dificil or ()):
            graus = graus + ["Difícil"]
        permitidas = list(DIFICULDADES) if incluir_dif_ausente or not graus else [d for d in DIFICULDADES
                                                                                   if d in graus]
        fpd = h.get("falta_por_dificuldade") or {}
        falta_dif = {d: int(fpd.get(d, 0)) for d in permitidas}
        fora = {d: v for d, v in fpd.items() if d not in permitidas and v > 0}
        deficit = max(int(h.get("meta", 0)) - int(h.get("n_efetivas", 0)), sum(falta_dif.values()),
                      sum((h.get("falta_por_subtema") or {}).values()), 0)
        if fora:
            ignoradas.append({"ano": ano, "habilidade": hab, "motivo": "dificuldade_ausente_no_banco",
                              "falta_ignorada": fora})
        if deficit <= 0:
            continue
        alvo = min(deficit, meta_por_hab) if meta_por_hab else deficit
        lista.append({"ano": ano, "habilidade": hab, "descricao": h.get("descricao", ""),
                      "meta": h.get("meta", 0), "deficit": deficit, "alvo": alvo,
                      "permitidas": permitidas, "falta_dif": falta_dif,
                      "falta_sub": dict(h.get("falta_por_subtema") or {}),
                      "fracao": deficit / max(1, int(h.get("meta", 1)))})
    ordem = {a: i for i, a in enumerate(ORDEM_ANOS)}
    lista.sort(key=lambda x: (-x["fracao"], ordem.get(x["ano"], 99), x["habilidade"]))
    return lista, ignoradas


def dificuldades_para(lac, n, ja_aceitas, seed=0):
    """n dificuldades priorizando o que falta (descontado o já injetado);
    escolher sempre a de maior falta intercala naturalmente F/M/D.

    Empate (T6, revisão do piloto): antes o desempate era a ordem F, M, D, e
    toda habilidade com faltas iguais começava — e, com orçamento curto,
    terminava — em Fácil. Agora o empate é sorteado com semente fixa por
    (habilidade, já aceitas, posição): determinístico, mas sem favorecer
    nenhuma dificuldade. Sem falta nenhuma, gira a partir de um início
    sorteado do mesmo jeito."""
    perm = lac["permitidas"]
    falta = {d: max(0, lac["falta_dif"].get(d, 0) - ja_aceitas.get(d, 0)) for d in perm}
    base = f"{seed}|{lac.get('ano')}|{lac.get('habilidade')}|{sum(ja_aceitas.values())}"
    inicio = random.Random(base + "|giro").randrange(len(perm)) if perm else 0
    out = []
    for i in range(n):
        if any(falta.values()):
            maior = max(falta.values())
            empatadas = [d for d in perm if falta[d] == maior]
            d = random.Random(f"{base}|{i}").choice(empatadas) if len(empatadas) > 1 else empatadas[0]
            falta[d] -= 1
        else:
            d = perm[(inicio + i) % len(perm)]
        out.append(d)
    return out


def planejar_slots(lac, n, ja_aceitas_dif, historico, taxonomia, seed):
    """Lista de (dificuldade, slot|None) para n questões. historico: questões já
    existentes (base + injetadas) da habilidade, para o planejador começar
    pelos subtemas menos cobertos. Sem taxonomia para a habilidade, slot None
    (prompt sem sufixo, igual ao distill_teacher)."""
    ano, hab = lac["ano"], lac["habilidade"]
    difs = dificuldades_para(lac, n, ja_aceitas_dif, seed=seed)
    if diversidade.obter_habilidade(ano, hab, taxonomia) is None:
        return [(d, None) for d in difs]
    hist = [diversidade.classificar_questao(q, ano, hab, taxonomia) for q in historico]
    por_dif = {}
    for d in lac["permitidas"]:
        qtd = difs.count(d)
        if not qtd:
            continue
        slots = diversidade.planejar_lote(ano, hab, qtd, dificuldade=d, seed=seed, historico=hist,
                                          taxonomia=taxonomia)
        por_dif[d] = list(slots)
        # o histórico cresce com o plano: a próxima dificuldade continua a rotação
        hist = hist + [{"subtema": s["subtema"], "contexto": s["contexto"]} for s in slots]
    return [(d, por_dif[d].pop(0)) for d in difs]


# ---------------------------------------------------------------------------
# Estado (retomada)
# ---------------------------------------------------------------------------

_COD_INJ = re.compile(r'"codigo_item"\s*:\s*"(INJ-[^"]+)"')


def caminho_pendente(saida):
    """Diário (write-ahead) da aceita que está sendo gravada."""
    return Path(str(saida) + ".pendente")


class SaidaOcupada(RuntimeError):
    """Outro processo já escreve nesta saída."""


@contextlib.contextmanager
def trava_saida(saida):
    """Trava EXCLUSIVA (flock) de `<saida>.lock` enquanto este processo lê o
    estado e escreve na saída. Revisão do piloto 2: dois processos de injeção
    na mesma saída calculavam o mesmo próximo codigo_item e o 2º abortava no
    unlink do diário .pendente compartilhado. Agora o 2º para ANTES de gastar."""
    import fcntl
    lock = aq._garantir_gravavel(Path(str(saida) + ".lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SaidaOcupada(f"outro processo já escreve em {Path(saida).name} ({lock.name} travado)") from None
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def caminho_truncadas(saida):
    """Quarentena dos fragmentos de linha truncada (nunca apagados)."""
    saida = Path(saida)
    return saida.with_name(saida.stem + "_truncadas.jsonl")


def _gravar_atomico(path, texto):
    path = aq._garantir_gravavel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(texto)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def gravar_aceita(f_ok, saida, ex):
    """Grava uma aceita (questão PAGA) sem poder perdê-la numa interrupção.

    T3 (revisão do piloto): 1) o registro completo vai para o diário
    `<saida>.pendente` de forma atômica (tmp + fsync + rename); 2) a linha é
    anexada à saída com flush + fsync; 3) o diário é apagado. Se o processo
    morre no passo 2, a linha fica truncada, mas o diário está inteiro e
    reparar_saida() o recupera na próxima execução."""
    linha = json.dumps(ex, ensure_ascii=False)
    _gravar_atomico(caminho_pendente(saida), linha)
    f_ok.write(linha + "\n")
    f_ok.flush()
    os.fsync(f_ok.fileno())
    caminho_pendente(saida).unlink()


def reparar_saida(saida):
    """Realinha a saída ANTES de qualquer append (T3). Devolve um relatório.

    - linha inválida (truncada) sai do arquivo e vai para a quarentena
      `<stem>_truncadas.jsonl`, com o codigo_item que der para ler dela (esse
      código fica RESERVADO: maior_sufixo nunca o reutiliza);
    - diário `.pendente` cujo codigo_item não está entre as linhas válidas é
      a aceita que a interrupção cortou: volta INTEIRA para a saída (a questão
      paga não se perde e o código não se duplica);
    - o arquivo é reescrito de forma atômica e termina em "\\n"."""
    saida = Path(saida)
    pend = caminho_pendente(saida)
    rel = {"fragmentos": 0, "recuperada": None, "codigos_reservados": []}
    texto = saida.read_text(encoding="utf-8") if saida.exists() else ""
    validas, fragmentos = [], []
    for linha in texto.splitlines():
        if not linha.strip():
            continue
        try:
            json.loads(linha)
            validas.append(linha)
        except json.JSONDecodeError:
            fragmentos.append(linha)
    mudou = bool(fragmentos) or (bool(texto) and not texto.endswith("\n"))
    if pend.exists():
        try:
            pendente = json.loads(pend.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pendente = None  # gravado de forma atômica: só se alguém editou à mão
        codigos = {json.loads(l).get("meta", {}).get("codigo_item") for l in validas}
        cod = (pendente or {}).get("meta", {}).get("codigo_item")
        if pendente is not None and cod not in codigos:
            validas.append(json.dumps(pendente, ensure_ascii=False))
            rel["recuperada"] = cod
            mudou = True
    if fragmentos:
        with aq.abrir_para_gravar(caminho_truncadas(saida), "a") as f:
            for frag in fragmentos:
                m = _COD_INJ.search(frag)
                cod = m.group(1) if m else None
                if cod:
                    rel["codigos_reservados"].append(cod)
                f.write(json.dumps({"ts": aq.agora_iso(), "arquivo": saida.name, "codigo_item": cod,
                                    "fragmento": frag}, ensure_ascii=False) + "\n")
        rel["fragmentos"] = len(fragmentos)
    if mudou:
        _gravar_atomico(saida, "".join(l + "\n" for l in validas))
    if pend.exists():
        pend.unlink()
    return rel


def estado_previo(saida, rejeitadas):
    """Por (ano, hab): aceitas (questões), por dificuldade, nº de tentativas já
    gastas (aceitas + rejeitadas que chegaram a gerar algo)."""
    est = defaultdict(lambda: {"aceitas": [], "dif": Counter(), "tentativas": 0})
    letras = Counter()
    exemplos = carregar_jsonl(saida)
    for ex in exemplos:
        m = ex.get("meta", {})
        chave = (m.get("ano"), m.get("habilidade"))
        try:
            q = extract_questao(json.loads(ex["messages"][2]["content"]), 0)
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if q:
            est[chave]["aceitas"].append(q)
            est[chave]["dif"][m.get("dificuldade")] += 1
            est[chave]["tentativas"] += 1
            letras[q.get("resposta_correta")] += 1
    rejs = carregar_jsonl(rejeitadas)
    for r in rejs:
        est[(r.get("ano"), r.get("habilidade"))]["tentativas"] += 1
    quarentena = carregar_jsonl(caminho_truncadas(saida))
    return est, letras, len(exemplos), maior_sufixo(exemplos, rejs + quarentena)


def maior_sufixo(exemplos, rejeitadas=()):
    """Maior número final de codigo_item "INJ-...-NNNNN" já usado (aceitas,
    aceitas movidas depois para as rejeitadas e códigos lidos de fragmentos
    truncados em quarentena).

    Por que não a contagem de linhas: depois que uma aceita sai do arquivo
    (auditoria independente) ou de uma linha truncada, a contagem fica menor
    que o maior código e a próxima aceita repetiria um codigo_item — e o
    --montar descarta a repetida como duplicata (questão paga perdida)."""
    maior = 0
    cods = [ex.get("meta", {}).get("codigo_item") for ex in exemplos] + [r.get("codigo_item") for r in rejeitadas]
    for c in cods:
        m = re.match(r"INJ-.*-(\d+)$", str(c or ""))
        if m:
            maior = max(maior, int(m.group(1)))
    return maior


def _voto_rev2(juizo):
    """{} sem 2º revisor ligado; {"revisor2": True} se aprovou; {"revisor2": None,
    "revisor2_ausente": motivo} se o Gemini estava indisponível (a questão entrou só com
    validador+revisor e pode ser reauditada depois)."""
    r2 = juizo.get("revisor2")
    if r2 is None:
        return {}
    if r2.get("indisponivel"):
        return {"revisor2": None, "revisor2_ausente": str(r2.get("motivo") or "indisponivel")[:120]}
    return {"revisor2": True}


def _ligar_gemini(args, agentes):
    """--segundo-revisor-gemini: Gemini como 2º revisor, com orçamento próprio."""
    if not getattr(args, "segundo_revisor_gemini", False):
        return
    if not args.dry_run and getattr(args, "max_chamadas_gemini", None) is None:
        raise SystemExit("--max-chamadas-gemini é obrigatório com --segundo-revisor-gemini "
                         "(as chamadas custam dinheiro).")
    import arbitro_gemini as ag
    ag.ligar_segundo_revisor(agentes, dry_run=args.dry_run,
                             max_chamadas=args.max_chamadas_gemini or ag.TETO_PADRAO_CHAMADAS,
                             modelo=getattr(args, "modelo_gemini", None))


def escolher_letra(letras, rng):
    """Letra menos usada entre as aceitas (empate sorteado): o gabarito tende ao
    uniforme sem mexer na questão depois de gerada (permutar depois quebraria
    resoluções que citam letras)."""
    menor = min(letras.get(L, 0) for L in aq.LETRAS)
    return rng.choice([L for L in aq.LETRAS if letras.get(L, 0) == menor])


def construir_exemplo(lac, descricao, dificuldade, questao, slot, idx, juizo, modelos, dificuldade_pedida=None):
    """Exemplo no formato do treino. `dificuldade` é a FINAL; quando difere da
    pedida (D3: o revisor julgou Fácil uma candidata pedida como Moderado/
    Difícil e os dois juízes aprovaram), o rótulo Fácil vai CONSISTENTE para
    os três lugares que o modelo vê/aprende: meta.dificuldade, "difficulty"
    da questão (EASY) e "Dificuldade: Fácil." do prompt de usuário."""
    ano, hab = lac["ano"], lac["habilidade"]
    val, rev = juizo["validador"], juizo["revisor"]
    pedida = dificuldade_pedida or dificuldade
    if dificuldade != pedida:
        if dificuldade != "Fácil":
            raise ValueError("D3 só rebaixa para Fácil")
        questao = dict(questao, difficulty=DIFFICULTY_MAP["Fácil"])
    confs = [c for c in (val.get("confianca"), rev.get("confianca")) if c is not None]
    ano_num = re.sub(r"\D", "", ano)  # código ASCII: "INJ-9-H17-M-00001"
    meta = {
        "codigo_item": f"INJ-{ano_num}-{hab}-{SIGLA_DIF.get(dificuldade, 'X')}-{idx:05d}",
        "ano": ano, "habilidade": hab, "dificuldade": dificuldade, "dificuldade_pedida": pedida,
        "dificuldade_real": rev.get("dificuldade_real"),
        # destilado=True: relatorio_base/merge tratam como gerado por professor
        # (não "real"); origem distingue este lote dos destilados antigos.
        "destilado": True, "origem": "injecao_saeb", "professor": modelos["gerador"],
        "vereditos": dict({"validador": True, "revisor": True}, **_voto_rev2(juizo)),
        "confianca": dict({"nivel": "alta", "validador": val.get("confianca"), "revisor": rev.get("confianca"),
                           "min": min(confs) if confs else None},
                          **({"revisor2": juizo["revisor2"].get("confianca")}
                             if (juizo.get("revisor2") or {}).get("veredito") else {})),
        "modelos": dict(modelos), "versao_prompts": aq.VERSAO_PROMPTS, "versao_juizes": aq.VERSAO_JUIZES,
        "data": aq.agora_iso(),
        "hash_questao": aq.hash_questao(questao),
    }
    if dificuldade != pedida:
        meta["rotulo_corrigido"] = {"de": pedida, "para": dificuldade, "fonte": "revisor:dificuldade_real",
                                    "versao_prompts": aq.VERSAO_PROMPTS}
    if slot:
        meta.update({"subtema": slot["subtema"], "tipo_raciocinio": slot["tipo_raciocinio"],
                     "contexto": slot["contexto"]})
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            # MESMO formato de inferência: USER_TEMPLATE + sufixo de diversidade
            {"role": "user", "content": dt.prompt_usuario(ano, hab, descricao, dificuldade, slot)},
            {"role": "assistant", "content": json.dumps({"questoes": [questao]}, ensure_ascii=False)},
        ],
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Laço principal
# ---------------------------------------------------------------------------

def caminhos(args):
    if args.dry_run:
        base = Path(args.dir_dry_run)
        return {"saida": Path(args.saida or base / "injecao_saeb.jsonl"),
                "rejeitadas": Path(args.rejeitadas or base / "injecao_rejeitadas.jsonl"),
                "resumo": Path(args.resumo or base / "injecao_resumo.json"),
                "uso": Path(args.log_uso or base / "uso_api.jsonl")}
    return {"saida": Path(args.saida or SAIDA_PADRAO), "rejeitadas": Path(args.rejeitadas or REJEITADAS_PADRAO),
            "resumo": Path(args.resumo or RESUMO_PADRAO), "uso": Path(args.log_uso or aq.LOG_USO_PADRAO)}


def _registro_rejeicao(lac, dificuldade, slot, tentativa, etapa, motivo, questao=None, bruto=None,
                       detalhes=None, juizo=None, modelos=None):
    r = {"ts": aq.agora_iso(), "ano": lac["ano"], "habilidade": lac["habilidade"], "dificuldade": dificuldade,
         "subtema": (slot or {}).get("subtema"), "tipo_raciocinio": (slot or {}).get("tipo_raciocinio"),
         "contexto": (slot or {}).get("contexto"), "tentativa": tentativa, "etapa": etapa, "motivo": motivo,
         "versao_prompts": aq.VERSAO_PROMPTS, "modelos": modelos}
    if questao is not None:
        r["questao"] = questao
    if bruto:
        r["bruto"] = aq._sanitizar(bruto, 600)
    if detalhes:
        r["detalhes"] = detalhes
    if juizo:
        for papel in ("validador", "revisor"):
            j = juizo.get(papel)
            if j:
                r[papel] = {k: j.get(k) for k in ("veredito", "avaliado", "resposta_calculada", "problemas",
                                                  "confianca", "criterios_falhos", "erro", "fases",
                                                  "dificuldade_real")
                            if k in j}
    return r


def injetar(args, agentes=None):
    """Executa a injeção. `agentes` permite injetar um Agentes com cliente falso
    (testes). Devolve o resumo (também gravado em JSON). Um processo por
    saída (trava_saida)."""
    p = caminhos(args)
    aq._garantir_gravavel(p["saida"])
    with trava_saida(p["saida"]):
        return _injetar(args, p, agentes)


def _injetar(args, p, agentes=None):
    for k in ("saida", "rejeitadas", "resumo", "uso"):
        aq._garantir_gravavel(p[k])
    if not args.dry_run and args.max_chamadas is None and agentes is None:
        raise SystemExit("--max-chamadas é obrigatório fora do --dry-run (as chamadas custam dinheiro).")
    if args.meta_relativa and not args.meta_por_habilidade:
        raise SystemExit("--meta-relativa exige --meta-por-habilidade.")
    regra_d3 = getattr(args, "regra_d3", None) or "suspensa"
    relatorio = json.loads(Path(args.relatorio).read_text(encoding="utf-8"))
    anos = {norm_ano(a) for a in args.anos.split(",")} if args.anos else None
    habs = parse_habilidades(args.habilidades) if args.habilidades else None
    lista, ignoradas = lacunas(relatorio, anos, habs, args.meta_por_habilidade, args.incluir_dificuldade_ausente)

    base_ex = carregar_jsonl(args.train)
    descr_treino = descricoes_treino(base_ex)
    base_qs = dt.historico_treino(base_ex)
    val_qs = dt.historico_treino([ex for vp in args.val for ex in carregar_jsonl(vp)])
    vistos = {dt.normalizar(q.get("enunciado", "")) for qs in list(base_qs.values()) + list(val_qs.values())
              for q in qs}
    # T3: realinha a saída (linha truncada -> quarentena; diário .pendente ->
    # aceita recuperada) ANTES de ler o estado e de qualquer append.
    reparo = reparar_saida(p["saida"])
    if reparo["fragmentos"] or reparo["recuperada"]:
        print(f"  saída reparada: {reparo['fragmentos']} fragmento(s) truncado(s) em "
              f"{caminho_truncadas(p['saida']).name}; aceita recuperada do diário: {reparo['recuperada']}")
    est, letras, n_existentes, ultimo_idx = estado_previo(p["saida"], p["rejeitadas"])
    for e in est.values():
        vistos.update(dt.normalizar(q.get("enunciado", "")) for q in e["aceitas"])
    # T1: índices de duplicata SEMÂNTICA (mesmos números + mesma resposta) de
    # TODA a base, da validação e do que já foi aceito.
    assin_base, assin_val = {}, {}
    for indice, grupos, rotulo in ((assin_base, base_qs, "base"), (assin_val, val_qs, "val")):
        for (a_, h_), qs in grupos.items():
            for q in qs:
                s_ = aq.assinatura_dados(q)
                if s_ is not None:
                    indice.setdefault(s_, f"{rotulo} {a_} {h_}: {str(q.get('enunciado', ''))[:80]}")
    for (a_, h_), e in est.items():
        for q in e["aceitas"]:
            s_ = aq.assinatura_dados(q)
            if s_ is not None:
                assin_base.setdefault(s_, f"injecao {a_} {h_}: {str(q.get('enunciado', ''))[:80]}")
    # --taxonomia só para testes/experimentos: o arquivo padrão está em edição
    # pelo outro workflow e é lido aqui SOMENTE para planejar slots.
    tax = diversidade.carregar_taxonomia(args.taxonomia)

    if agentes is None:
        agentes = aq.montar_agentes(args, simulado=args.dry_run, log_uso=p["uso"])
        _ligar_gemini(args, agentes)
    if getattr(args, "cache_juizes", None) and getattr(agentes, "cache_juizes", None) is None:
        # Guarda as respostas PAGAS dos juízes (permite reavaliar a decisão
        # depois sem pagar de novo). Candidatos são sempre novos: não há acerto.
        agentes.cache_juizes = aq.CacheRespostas(args.cache_juizes)
    modelos = dict(agentes.modelos)
    rng = random.Random(f"letra|{args.seed}|{n_existentes}")

    # Fila de slots por habilidade (só o que falta, descontado o já aceito)
    filas, ctrl = {}, {}
    for lac in lista:
        chave = (lac["ano"], lac["habilidade"])
        e = est[chave]
        if args.meta_relativa:
            # Nova rodada sobre uma habilidade que já tem aceitas: a meta conta a
            # partir delas (sem passar do déficit) e o teto de tentativas também.
            lac["alvo"] = min(lac["deficit"], len(e["aceitas"]) + args.meta_por_habilidade)
            teto = e["tentativas"] + max(0, lac["alvo"] - len(e["aceitas"])) * args.tentativas_por_slot
        else:
            teto = lac["alvo"] * args.tentativas_por_slot
        restante = lac["alvo"] - len(e["aceitas"])
        if restante <= 0 or e["tentativas"] >= teto:
            continue
        hist = base_qs.get(chave, []) + e["aceitas"]
        slots = planejar_slots(lac, restante, e["dif"], hist, tax, seed=args.seed * 1000 + e["tentativas"])
        filas[chave] = [{"dif": d, "slot": s, "tent": 0} for d, s in slots]
        # T4: respostas não numéricas já usadas pelas aceitas desta habilidade
        respostas, textos = Counter(), {}
        for q in e["aceitas"]:
            rc = aq.resposta_conteudo(q)
            if rc:
                respostas[rc] += 1
                textos.setdefault(rc, str(q["alternativas"][q["resposta_correta"]]))
        ctrl[chave] = {"lac": lac, "descricao": descr_treino.get(chave) or lac["descricao"],
                       "teto": teto, "tentativas": e["tentativas"], "aceitas_previas": len(e["aceitas"]),
                       "aceitas": 0, "rejeitadas": Counter(), "rebaixadas": 0, "respostas": respostas,
                       "dif_candidatos": Counter(), "dif_aceitas": Counter(), "dif_final": Counter(),
                       "textos_resposta": textos,
                       "limite_resposta": max(2, math.ceil(FRACAO_MAX_MESMA_RESPOSTA * lac["alvo"]))}

    print(f"Injeção ({'DRY-RUN, cliente simulado' if args.dry_run else 'API REAL'}) — "
          f"{len(filas)} habilidades com déficit a atacar; orçamento: {args.max_chamadas} chamadas")
    print(f"  saída: {p['saida']}\n  rejeitadas: {p['rejeitadas']}\n  versão dos prompts: {aq.VERSAO_PROMPTS}")
    print(f"  verificador de geometria: {'disponível' if aq._vg is not None else 'indisponível (segue sem ele)'}")

    etapas = Counter()
    motivos = Counter()
    parada = None
    t_ini = time.monotonic()  # tempo de parede: entra no custo por aceita
    erros_api_seguidos = 0
    idx = max(n_existentes, ultimo_idx)
    ativos = [k for k in filas]
    with aq.abrir_para_gravar(p["saida"], "a") as f_ok, aq.abrir_para_gravar(p["rejeitadas"], "a") as f_rej:
        while ativos and parada is None:
            # Rodada: cada habilidade ativa tenta UM slot até aceitar ou esgotar as
            # tentativas — o orçamento se espalha entre habilidades em vez de
            # acabar nas primeiras da lista.
            for chave in list(ativos):
                c = ctrl[chave]
                fila = filas[chave]
                lac = c["lac"]
                while fila:
                    item = fila[0]
                    if c["tentativas"] >= c["teto"]:
                        fila.clear()
                        break
                    if not agentes.orcamento.cabe(CHAMADAS_PIOR_CASO):
                        parada = "orcamento_insuficiente_para_candidato"
                        break
                    item["tent"] += 1
                    c["tentativas"] += 1
                    ctx = {"assinaturas_base": assin_base, "assinaturas_val": assin_val,
                           "respostas_usadas": c["respostas"], "limite_resposta": c["limite_resposta"],
                           "evitar_respostas": respostas_a_evitar(c),
                           # D3: há um slot Fácil pendente que uma rebaixada pode ocupar?
                           "facil_aberto": any(it["dif"] == "Fácil" for it in fila[1:]),
                           "regra_d3": regra_d3}
                    try:
                        res = tentar(agentes, lac, c["descricao"], item, letras, rng, vistos,
                                     base_qs.get(chave, []) + est[chave]["aceitas"], val_qs.get(chave, []), tax,
                                     **ctx)
                    except aq.OrcamentoEsgotado as exc:
                        parada = f"orcamento_esgotado: {exc}"
                        c["tentativas"] -= 1  # nada foi gravado; a retomada refaz
                        item["tent"] -= 1
                        break
                    except KeyboardInterrupt:
                        # Ctrl+C/SIGINT (ex.: remanejar orçamento no meio do piloto): o
                        # candidato em andamento se perde, mas o resumo com o funil e o
                        # uso do que JÁ foi pago ainda é gravado. A chamada interrompida
                        # pode ter sido cobrada sem aparecer no log de uso.
                        parada = "interrompido (SIGINT); a chamada em andamento pode ter sido cobrada"
                        c["tentativas"] -= 1
                        item["tent"] -= 1
                        break
                    etapas[res["etapa"]] += 1
                    c["dif_candidatos"][item["dif"]] += 1
                    if res["aceita"]:
                        idx += 1
                        dif_final = res.get("dif_final") or item["dif"]
                        c["dif_aceitas"][item["dif"]] += 1
                        c["dif_final"][dif_final] += 1
                        ex = construir_exemplo(lac, c["descricao"], dif_final, res["questao"], item["slot"],
                                               idx, res["juizo"], modelos, dificuldade_pedida=item["dif"])
                        gravar_aceita(f_ok, p["saida"], ex)
                        # SÓ AGORA o estado de dedup é atualizado (veredito duplo)
                        q_ok = res["questao"]
                        est[chave]["aceitas"].append(q_ok)
                        vistos.add(dt.normalizar(q_ok.get("enunciado", "")))
                        s_ = aq.assinatura_dados(q_ok)
                        if s_ is not None:
                            assin_base.setdefault(s_, f"injecao {ex['meta']['codigo_item']}")
                        rc = aq.resposta_conteudo(q_ok)
                        if rc:
                            c["respostas"][rc] += 1
                            c["textos_resposta"].setdefault(rc, str(q_ok["alternativas"][q_ok["resposta_correta"]]))
                        letras[q_ok.get("resposta_correta")] += 1
                        c["aceitas"] += 1
                        erros_api_seguidos = 0
                        if dif_final != item["dif"]:
                            # D3: entrou como Fácil e ocupa um slot Fácil pendente; o
                            # slot Moderado/Difícil continua aberto (sem gastar tentativa).
                            j = next(k for k in range(1, len(fila)) if fila[k]["dif"] == "Fácil")
                            fila.pop(j)
                            item["tent"] -= 1
                            c["rebaixadas"] += 1
                        else:
                            fila.pop(0)
                        print(f"  [+] {lac['ano']} {lac['habilidade']} {dif_final}"
                              f"{' (pedida ' + item['dif'] + ')' if dif_final != item['dif'] else ''} "
                              f"({(item['slot'] or {}).get('subtema', '-')}) aceita "
                              f"— {c['aceitas_previas'] + c['aceitas']}/{lac['alvo']} | "
                              f"chamadas {agentes.orcamento.chamadas}")
                        break
                    motivo = res["motivo"]
                    motivos[f"{res['etapa']}:{motivo}"] += 1
                    c["rejeitadas"][f"{res['etapa']}:{motivo}"] += 1
                    f_rej.write(json.dumps(_registro_rejeicao(
                        lac, item["dif"], item["slot"], item["tent"], res["etapa"], motivo,
                        questao=res.get("questao"), bruto=res.get("bruto"), detalhes=res.get("detalhes"),
                        juizo=res.get("juizo"), modelos=modelos), ensure_ascii=False) + "\n")
                    f_rej.flush()
                    erros_api_seguidos = erros_api_seguidos + 1 if motivo == "erro_api" else 0
                    if erros_api_seguidos >= args.max_erros_api:
                        parada = f"{erros_api_seguidos} erros de API seguidos"
                        break
                    if item["tent"] >= args.tentativas_por_slot:
                        fila.pop(0)  # desiste do slot; o próximo traz outro subtema/contexto
                    else:
                        continue
                    break
                if parada:
                    break
                if not fila:
                    ativos.remove(chave)

    resumo = _resumo(args, p, agentes, ctrl, etapas, motivos, ignoradas, parada, lista)
    tempo = round(time.monotonic() - t_ini, 1)
    resumo["tempo_s"] = tempo
    resumo["custo_por_aceita"]["tempo_s"] = (round(tempo / resumo["funil"]["aceitos"], 1)
                                             if resumo["funil"]["aceitos"] else None)
    aq._garantir_gravavel(p["resumo"])
    p["resumo"].parent.mkdir(parents=True, exist_ok=True)
    p["resumo"].write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    _imprimir_resumo(resumo)
    return resumo


def respostas_a_evitar(c, maximo=5):
    """T4: textos das respostas não numéricas que já estão no limite (ou a uma
    aceita dele) nesta habilidade — vão no prompt do professor."""
    perto = max(1, c["limite_resposta"] - 1)
    usadas = [rc for rc, n in c["respostas"].most_common() if n >= perto]
    return [c["textos_resposta"].get(rc, rc) for rc in usadas[:maximo]]


def motivo_revisor(rev):
    """Motivo registrado de uma reprovação do revisor: o 1º problema que VETA.

    Aviso (dificuldade_incoerente) e incoerencia_interna não vetam sozinhos;
    no piloto 2, 5 rejeições por resposta_nao_unica/distrator_verdadeiro
    saíram rotuladas como "dificuldade_incoerente" (só a contagem estava
    errada; a decisão de reprovar estava certa)."""
    prob = rev.get("problemas") or []
    falhos = rev.get("criterios_bloqueantes_falhos") or []
    bloq = [p["codigo"] for p in prob if p["codigo"] not in aq.CODIGOS_INFORMATIVOS | {"incoerencia_interna"}]
    if bloq:
        return bloq[0]
    if falhos:
        return f"criterio_{falhos[0]}"
    return prob[0]["codigo"] if prob else "reprovado"


def tentar(agentes, lac, descricao, item, letras, rng, vistos, comparar_com, val_questoes, tax,
           assinaturas_base=None, assinaturas_val=None, respostas_usadas=None, limite_resposta=None,
           evitar_respostas=(), facil_aberto=False, regra_d3="suspensa"):
    """Um candidato: gerar -> filtros -> validador -> revisor. Devolve
    {aceita, etapa, motivo, questao?, juizo?, dif_final?}.

    D3: aprovada pelos dois juízes, mas pedida como Moderado/Difícil e julgada
    Fácil pelo revisor -> entra como Fácil se `facil_aberto` (há slot Fácil
    pendente na habilidade); senão é rejeitada (entrar com o rótulo pedido
    ensinaria um rótulo que o revisor diz estar errado).

    Passo 2 (2026-10-01): isso só vale com regra_d3="revisor". Com a regra
    fixada ("suspensa"), a aprovada entra com a dificuldade pedida: rejeitar
    por um sinal que erra 83% dos M/D reais jogaria fora candidatas corretas
    (e o custo já pago nelas)."""
    ano, hab, dif, slot = lac["ano"], lac["habilidade"], item["dif"], item["slot"]
    letra = escolher_letra(letras, rng)
    g = agentes.gerar(ano, hab, descricao, dif, slot, letra_alvo=letra, evitar_respostas=evitar_respostas)
    if g["erro"] == "erro_api":
        return {"aceita": False, "etapa": "geracao", "motivo": "erro_api",
                "detalhes": {"erro": (g.get("info") or {}).get("erro")}}
    motivo, detalhes = aq.filtro_injecao(g["obj"], g["texto"] or "", ano=ano, habilidade=hab, dificuldade=dif,
                                         slot=slot, vistos=vistos, comparar_com=comparar_com,
                                         val_questoes=val_questoes, taxonomia=tax, assinaturas_base=assinaturas_base,
                                         assinaturas_val=assinaturas_val, respostas_usadas=respostas_usadas,
                                         limite_resposta=limite_resposta)
    if motivo:
        return {"aceita": False, "etapa": "filtro", "motivo": motivo, "questao": g["questao"],
                "bruto": None if g["questao"] else g["texto"], "detalhes": detalhes}
    q = g["questao"]
    juizo = agentes.julgar(q, ano, hab, descricao, (slot or {}).get("subtema_rotulo"), dif, curto_circuito=True)
    if not juizo["validador"]["veredito"]:
        prob = juizo["validador"].get("problemas") or [{"codigo": "reprovado"}]
        return {"aceita": False, "etapa": "validador", "motivo": prob[-1]["codigo"] if prob else "reprovado",
                "questao": q, "juizo": juizo}
    if not juizo["revisor"]["veredito"]:
        mot = motivo_revisor(juizo["revisor"])
        return {"aceita": False, "etapa": "revisor", "motivo": mot, "questao": q, "juizo": juizo}
    rev2 = juizo.get("revisor2")
    if rev2 is not None and not rev2.get("indisponivel") and not rev2["veredito"]:
        # 2º revisor (Gemini). Falha de API/JSON malformado também reprova (fail-closed).
        return {"aceita": False, "etapa": "revisor2", "motivo": motivo_revisor(rev2), "questao": q,
                "juizo": juizo}
    if (regra_d3 == "revisor" and dif in ("Moderado", "Difícil")
            and juizo["revisor"].get("dificuldade_real") == "Fácil"):
        if not facil_aberto:
            return {"aceita": False, "etapa": "revisor", "motivo": "dificuldade_real_facil_sem_lacuna",
                    "questao": q, "juizo": juizo}
        return {"aceita": True, "etapa": "aceita", "motivo": None, "questao": q, "juizo": juizo,
                "dif_final": "Fácil"}
    return {"aceita": True, "etapa": "aceita", "motivo": None, "questao": q, "juizo": juizo, "dif_final": dif}


# ---------------------------------------------------------------------------
# Manutenção das aceitas: rejulgamento, remoção por auditoria, rótulo (D3)
# ---------------------------------------------------------------------------

_SUBTEMA_PROMPT = re.compile(r"Subtema: (.*?)\. Tipo de raciocínio")


def _questao_e_contexto(ex):
    """(questão, descrição, subtema_rotulo, dificuldade pedida) de uma aceita,
    reconstruídos do PRÓPRIO exemplo (mesmas strings que a injeção mandou ao
    revisor: descrição e subtema do prompt de treino)."""
    import auditar_base as ab
    q = extract_questao(json.loads(ex["messages"][2]["content"]), 0)
    user = next((m.get("content", "") for m in ex.get("messages", []) if m.get("role") == "user"), "")
    sub = _SUBTEMA_PROMPT.search(user)
    m = ex.get("meta", {})
    return q, ab.descricao_do_prompt(ex), sub.group(1) if sub else None, m.get("dificuldade_pedida") or m.get("dificuldade")


def filtros_rejulgamento(q, ano, habilidade):
    """Filtros determinísticos que valem para uma aceita JÁ gravada (sem os
    de duplicata contra si mesma). Devolve o motivo ou None."""
    d = aq.defeitos_resolucao(q)
    if d:
        return d[0]
    texto = " ".join([str(q.get("enunciado", ""))] + [str(v) for v in (q.get("alternativas") or {}).values()])
    if aq.depende_de_visual_ausente(texto):
        return "dependencia_visual"
    if aq.contexto_inverossimil(q):
        return "contexto_inverossimil"
    if aq.verificar_aritmetica(q)["status"] == "contradiz":
        return "verificador_aritmetico"
    if aq.veredito_geometria(q, ano, habilidade)["status"] == "reprovada":
        return "geometria"
    if aq.veredito_d5_geometria(q)["status"] == "reprovada":
        return "geometria_d5"  # P1: a mesma D5 determinística da injeção
    return None


def _registro_saida(ex, etapa, motivo, detalhe=None, juizo=None):
    """Registro de uma aceita que SAI da saída (vai para as rejeitadas). O
    codigo_item fica nas rejeitadas: maior_sufixo nunca o reutiliza."""
    m = ex.get("meta", {})
    try:
        q = extract_questao(json.loads(ex["messages"][2]["content"]), 0)
    except (KeyError, IndexError, ValueError, TypeError):
        q = None
    r = {"ts": aq.agora_iso(), "ano": m.get("ano"), "habilidade": m.get("habilidade"),
         "dificuldade": m.get("dificuldade"), "subtema": m.get("subtema"),
         "tipo_raciocinio": m.get("tipo_raciocinio"), "contexto": m.get("contexto"), "tentativa": None,
         "etapa": etapa, "motivo": motivo, "codigo_item": m.get("codigo_item"),
         "versao_prompts": m.get("versao_prompts"), "modelos": m.get("modelos"), "questao": q,
         "exemplo": ex}
    if detalhe:
        r["detalhe"] = detalhe
    if juizo:
        for papel in ("validador", "revisor"):
            j = juizo.get(papel)
            if j:
                r[papel] = {k: j.get(k) for k in ("veredito", "avaliado", "resposta_calculada", "problemas",
                                                  "confianca", "criterios_falhos", "erro", "fases",
                                                  "dificuldade_real", "status") if k in j}
    return r


def _reescrever(saida, rejeitadas, mantidas, saem):
    """Grava as que saem nas rejeitadas (append, fsync) e DEPOIS reescreve a
    saída de forma atômica: uma interrupção no meio deixa no máximo a aceita
    nos dois arquivos (o --montar só lê a saída), nunca em nenhum."""
    if saem:
        with aq.abrir_para_gravar(rejeitadas, "a") as f:
            for r in saem:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    _gravar_atomico(saida, "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in mantidas))


def mover_para_rejeitadas(args, notas, motivo="auditoria_independente"):
    """Tira da saída as aceitas de `notas` ({codigo_item: nota}) e as grava em
    rejeitadas com `motivo` (ex.: "auditoria_independente"). Códigos que não
    estão na saída são relatados, não ignorados em silêncio."""
    p = caminhos(args)
    with trava_saida(p["saida"]):
        reparar_saida(p["saida"])
        exemplos = carregar_jsonl(p["saida"])
        cods = {ex.get("meta", {}).get("codigo_item") for ex in exemplos}
        faltam = sorted(set(notas) - cods)
        saem = [_registro_saida(ex, "auditoria_independente", motivo, notas[ex["meta"]["codigo_item"]])
                for ex in exemplos if ex.get("meta", {}).get("codigo_item") in notas]
        mantidas = [ex for ex in exemplos if ex.get("meta", {}).get("codigo_item") not in notas]
        _reescrever(p["saida"], p["rejeitadas"], mantidas, saem)
    return {"movidas": [r["codigo_item"] for r in saem], "nao_encontradas": faltam, "restantes": len(mantidas)}


def rebaixar_aceitas(args, notas, fonte="auditoria_independente"):
    """D3 por decisão humana: rebaixa para Fácil, nos três lugares, as aceitas
    de `notas` ({codigo_item: nota}). Só Moderado/Difícil -> Fácil."""
    import auditar_base as ab
    p = caminhos(args)
    feitas, falhas = [], {}
    with trava_saida(p["saida"]):
        reparar_saida(p["saida"])
        exemplos = carregar_jsonl(p["saida"])
        novos = []
        for ex in exemplos:
            cod = ex.get("meta", {}).get("codigo_item")
            if cod not in notas:
                novos.append(ex)
                continue
            novo, falha = ab.rebaixar_para_facil(ex)
            if novo is None:
                falhas[cod] = falha
                novos.append(ex)
                continue
            novo["meta"]["rotulo_corrigido"] = dict(novo["meta"]["rotulo_corrigido"], fonte=fonte, nota=notas[cod],
                                                    versao_prompts=ex["meta"].get("versao_prompts"))
            novos.append(novo)
            feitas.append(cod)
        _reescrever(p["saida"], p["rejeitadas"], novos, [])
    return {"rebaixadas": feitas, "falhas": falhas, "nao_encontradas": sorted(set(notas) - set(feitas) - set(falhas))}


def precisa_rejulgar(ex, todas=False):
    m = ex.get("meta", {})
    if todas:
        return True
    return aq.VERSAO_PROMPTS not in (m.get("versao_prompts"), (m.get("rejulgamento") or {}).get("versao_prompts"))


def rejulgar(args, agentes=None):
    """Julga de novo, com os prompts e a decisão ATUAIS, as aceitas julgadas
    com outra versão de prompts (ou todas, --rejulgar-todas). Revisão do
    piloto 2: as 10 aceitas do piloto 1 entravam na v3 sem rejulgamento.

    - filtros determinísticos atuais + validador + revisor (curto-circuito);
    - aprovada pelos dois: fica, com meta.rejulgamento (versões, vereditos,
      dificuldade_real); com --regra-d3 revisor, julgada Fácil quando
      rotulada Moderado/Difícil é rebaixada para Fácil nos três lugares (D3;
      padrão "suspensa" desde o passo 2: nada é rebaixado);
    - reprovada (filtro ou juiz que AVALIOU): sai para as rejeitadas (etapa
      "rejulgamento");
    - não avaliada (erro de API/JSON malformado) ou sem orçamento: fica como
      está e continua fora da v3 até um novo rejulgamento.
    """
    import auditar_base as ab
    p = caminhos(args)
    regra_d3 = getattr(args, "regra_d3", None) or ab.REGRA_D3
    if not args.dry_run and args.max_chamadas is None and agentes is None:
        raise SystemExit("--max-chamadas é obrigatório fora do --dry-run (as chamadas custam dinheiro).")
    if agentes is None:
        agentes = aq.montar_agentes(args, simulado=args.dry_run, log_uso=p["uso"])
        _ligar_gemini(args, agentes)
    if getattr(args, "cache_juizes", None) and getattr(agentes, "cache_juizes", None) is None:
        agentes.cache_juizes = aq.CacheRespostas(args.cache_juizes)
    res = {"aprovadas": [], "rebaixadas": [], "saem": {}, "nao_avaliadas": [], "sem_orcamento": [],
           "ja_atuais": 0}
    with trava_saida(p["saida"]):
        reparar_saida(p["saida"])
        exemplos = carregar_jsonl(p["saida"])
        novos, saem = [], []
        for ex in exemplos:
            m = ex.get("meta", {})
            cod = m.get("codigo_item")
            if not precisa_rejulgar(ex, getattr(args, "rejulgar_todas", False)):
                res["ja_atuais"] += 1
                novos.append(ex)
                continue
            q, descricao, subtema, dif_pedida = _questao_e_contexto(ex)
            motivo = filtros_rejulgamento(q, m.get("ano"), m.get("habilidade"))
            if motivo:
                saem.append(_registro_saida(ex, "rejulgamento", f"filtro:{motivo}"))
                res["saem"][cod] = f"filtro:{motivo}"
                continue
            if not agentes.orcamento.cabe(3):
                res["sem_orcamento"].append(cod)
                novos.append(ex)
                continue
            try:
                j = agentes.julgar(q, m.get("ano"), m.get("habilidade"), descricao, subtema, dif_pedida,
                                   curto_circuito=True)
            except aq.OrcamentoEsgotado:
                res["sem_orcamento"].append(cod)
                novos.append(ex)
                continue
            val, rev = j["validador"], j["revisor"]
            if not val.get("avaliado") or (val["veredito"] and not (rev or {}).get("avaliado")):
                res["nao_avaliadas"].append(cod)
                novos.append(ex)
                continue
            if not j["ambos"]:
                rev2 = j.get("revisor2")
                papel = ("validador" if not val["veredito"] else "revisor" if not rev["veredito"] else "revisor2")
                mot = ((val.get("problemas") or [{"codigo": "reprovado"}])[-1]["codigo"] if papel == "validador"
                       else motivo_revisor(rev2 if papel == "revisor2" else rev))
                saem.append(_registro_saida(ex, "rejulgamento", f"{papel}:{mot}", juizo=j))
                res["saem"][cod] = f"{papel}:{mot}"
                continue
            rj = {"versao_prompts": aq.VERSAO_PROMPTS, "versao_juizes": aq.VERSAO_JUIZES, "data": aq.agora_iso(),
                  "modelos": dict(agentes.modelos), "vereditos": dict({"validador": True, "revisor": True}, **_voto_rev2(j)),
                  "confianca": dict({"validador": val.get("confianca"), "revisor": rev.get("confianca")},
                                    **({"revisor2": j["revisor2"].get("confianca")}
                                       if (j.get("revisor2") or {}).get("veredito") else {})),
                  "dificuldade_real": rev.get("dificuldade_real"), "de_versao_prompts": m.get("versao_prompts")}
            novo = dict(ex, meta=dict(m, rejulgamento=rj))
            if (regra_d3 == "revisor" and m.get("dificuldade") in ("Moderado", "Difícil")
                    and rev.get("dificuldade_real") == "Fácil"):
                reb, _falha = ab.rebaixar_para_facil(novo)
                if reb is not None:
                    reb["meta"]["rotulo_corrigido"] = dict(reb["meta"]["rotulo_corrigido"],
                                                           fonte="rejulgamento:revisor_dificuldade_real",
                                                           versao_prompts=aq.VERSAO_PROMPTS)
                    novo = reb
                    res["rebaixadas"].append(cod)
            novos.append(novo)
            res["aprovadas"].append(cod)
        _reescrever(p["saida"], p["rejeitadas"], novos, saem)
    res["uso"] = agentes.orcamento.resumo()
    if getattr(agentes, "cache_juizes", None) is not None:
        res["cache_juizes"] = agentes.cache_juizes.resumo()
    return res


def _resumo(args, p, agentes, ctrl, etapas, motivos, ignoradas, parada, lista):
    orc = agentes.orcamento.resumo()
    aceitas = etapas.get("aceita", 0)
    candidatos = sum(etapas.values())
    por_hab = {f"{k[0]} {k[1]}": {"alvo": c["lac"]["alvo"], "aceitas_antes": c["aceitas_previas"],
                                  "aceitas_agora": c["aceitas"], "tentativas_total": c["tentativas"],
                                  "rebaixadas_para_facil": c["rebaixadas"],
                                  "candidatos_por_dificuldade_pedida": dict(c["dif_candidatos"]),
                                  "aceitas_por_dificuldade_pedida": dict(c["dif_aceitas"]),
                                  "aceitas_por_dificuldade_final": dict(c["dif_final"]),
                                  "respostas_nao_numericas": dict(c["respostas"]),
                                  "rejeicoes": dict(c["rejeitadas"])} for k, c in ctrl.items()}
    funil = {"candidatos": candidatos,
             "rejeitados_geracao": etapas.get("geracao", 0), "rejeitados_filtro": etapas.get("filtro", 0),
             "rejeitados_validador": etapas.get("validador", 0), "rejeitados_revisor": etapas.get("revisor", 0),
             "rejeitados_revisor2": etapas.get("revisor2", 0),
             "aceitos": aceitas, "taxa_aceite": round(aceitas / candidatos, 3) if candidatos else None,
             "aceitos_rebaixados_para_facil": sum(c["rebaixadas"] for c in ctrl.values())}
    return {"gerado_em": aq.agora_iso(), "dry_run": bool(args.dry_run), "versao_prompts": aq.VERSAO_PROMPTS,
            "modelos": agentes.modelos, "arquivos": {k: str(v) for k, v in p.items()},
            "parametros": {"anos": args.anos, "habilidades": args.habilidades,
                           "meta_por_habilidade": args.meta_por_habilidade, "meta_relativa": args.meta_relativa,
                           "tentativas_por_slot": args.tentativas_por_slot, "max_chamadas": args.max_chamadas,
                           "incluir_dificuldade_ausente": args.incluir_dificuldade_ausente,
                           "dificil_liberado_nos_anos": list(ANOS_DIFICIL_LIBERADO)},
            "habilidades_com_deficit": len(lista), "habilidades_atacadas": len(ctrl),
            "geometria": "disponivel" if aq._vg is not None else "indisponivel",
            "funil": funil, "motivos": dict(motivos.most_common()), "uso": orc,
            "custo_por_aceita": {"chamadas": round(orc["chamadas"] / aceitas, 2) if aceitas else None,
                                 "tokens": round((orc["prompt_tokens"] + orc["completion_tokens"]) / aceitas)
                                 if aceitas else None},
            "parada": parada or "concluido", "ignoradas": ignoradas, "por_habilidade": por_hab}


def _imprimir_resumo(r):
    f = r["funil"]
    print("\n=== Resumo da injeção ===")
    print(f"Candidatos: {f['candidatos']} | aceitos: {f['aceitos']} | taxa: {f['taxa_aceite']}")
    print(f"Rejeitados — geração: {f['rejeitados_geracao']}, filtro: {f['rejeitados_filtro']}, "
          f"validador: {f['rejeitados_validador']}, revisor: {f['rejeitados_revisor']}"
          + (f", revisor2: {f['rejeitados_revisor2']}" if f.get("rejeitados_revisor2") else ""))
    print(f"Chamadas: {r['uso']['chamadas']} (teto {r['uso']['max_chamadas']}) | tokens: "
          f"{r['uso']['prompt_tokens']} entrada + {r['uso']['completion_tokens']} saída")
    print(f"Aceitas rebaixadas para Fácil (D3): {f['aceitos_rebaixados_para_facil']}")
    print(f"Custo por aceita: {r['custo_por_aceita']}")
    print(f"Parada: {r['parada']}")
    if r["motivos"]:
        print("Motivos: " + ", ".join(f"{k}={v}" for k, v in list(r["motivos"].items())[:12]))
    if r["ignoradas"]:
        print(f"Falta ignorada por dificuldade ausente no banco: {len(r['ignoradas'])} habilidades "
              "(use --incluir-dificuldade-ausente para pedir mesmo assim)")


def construir_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relatorio", default=str(RELATORIO_PADRAO))
    ap.add_argument("--train", default=str(TRAIN_PADRAO), help="base lida (somente leitura)")
    ap.add_argument("--val", nargs="*", default=[str(v) for v in VAL_PADRAO])
    ap.add_argument("--habilidades", default=None, help="ex.: H17 ou 9º:H17,2º:H07 (sem ano = todos os anos)")
    ap.add_argument("--anos", default=None, help="ex.: 2º,5º")
    ap.add_argument("--meta-por-habilidade", type=int, default=None,
                    help="teto de questões ACEITAS por habilidade (padrão: o déficit inteiro)")
    ap.add_argument("--meta-relativa", action="store_true",
                    help="--meta-por-habilidade conta a partir das aceitas que a habilidade JÁ tem na saída "
                         "(nova rodada: +N por habilidade, sem passar do déficit)")
    ap.add_argument("--tentativas-por-slot", type=int, default=3)
    ap.add_argument("--max-erros-api", type=int, default=5, help="aborta após N erros de API seguidos")
    ap.add_argument("--segundo-revisor-gemini", action="store_true",
                    help="liga o Gemini como 2º revisor: a candidata só entra se validador E revisor E "
                         "2º revisor aprovarem (padrão: desligado)")
    ap.add_argument("--max-chamadas-gemini", type=int, default=None,
                    help="teto DURO de chamadas ao Gemini (obrigatório com --segundo-revisor-gemini, fora do dry-run)")
    ap.add_argument("--modelo-gemini", default=None, help="padrão: o modelo do árbitro (gemini-2.5-flash)")
    ap.add_argument("--incluir-dificuldade-ausente", action="store_true",
                    help="pede também dificuldades que o banco real nunca usa na habilidade (ex.: Difícil no 1º)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taxonomia", default=None, help="padrão: data/taxonomia_subtemas.json (somente leitura)")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--rejeitadas", default=None)
    ap.add_argument("--resumo", default=None)
    ap.add_argument("--log-uso", default=None)
    ap.add_argument("--dir-dry-run", default=str(DRYRUN_DIR))
    ap.add_argument("--regra-d3", choices=("suspensa", "revisor"), default="suspensa",
                    help="D3: 'suspensa' (padrão, passo 2: entra com a dificuldade pedida) ou 'revisor' "
                         "(julgada Fácil pelo revisor entra como Fácil num slot Fácil livre)")
    ap.add_argument("--cache-juizes", nargs="*", default=None,
                    help="arquivos .jsonl de respostas dos juízes; as novas vão para o 1º (ex.: "
                         "outputs/agentes/juizes_cache.jsonl)")
    man = ap.add_argument_group("manutenção das aceitas (não gera questões)")
    man.add_argument("--rejulgar", action="store_true",
                     help="julga de novo, com os prompts atuais, as aceitas julgadas com outra versão de prompts")
    man.add_argument("--rejulgar-todas", action="store_true", help="com --rejulgar: todas as aceitas")
    man.add_argument("--mover-json", default=None,
                     help='JSON {codigo_item: nota}: tira essas aceitas da saída para as rejeitadas')
    man.add_argument("--motivo-mover", default="auditoria_independente")
    man.add_argument("--rebaixar-facil-json", default=None,
                     help='JSON {codigo_item: nota}: rebaixa essas aceitas para Fácil nos três lugares (D3)')
    aq.adicionar_args_modelos(ap)
    return ap


def main(argv=None):
    args = construir_parser().parse_args(argv)
    if args.mover_json:
        notas = json.loads(Path(args.mover_json).read_text(encoding="utf-8"))
        print(json.dumps(mover_para_rejeitadas(args, notas, args.motivo_mover), ensure_ascii=False, indent=1))
        return 0
    if args.rebaixar_facil_json:
        notas = json.loads(Path(args.rebaixar_facil_json).read_text(encoding="utf-8"))
        print(json.dumps(rebaixar_aceitas(args, notas), ensure_ascii=False, indent=1))
        return 0
    if args.rejulgar:
        print(json.dumps(rejulgar(args), ensure_ascii=False, indent=1))
        return 0
    injetar(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
