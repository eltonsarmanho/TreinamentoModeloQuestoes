"""Destilação de dados: gera questões novas com um modelo PROFESSOR grande
(via Hugging Face Inference Providers) e aceita só as que passam nos filtros
determinísticos — o caminho da literatura para ampliar a capacidade de um
modelo pequeno além do que os ~300 exemplos reais ensinam.

Diferente de generate_synthetic.py (aritmética pura com resposta_correta
calculada em Python), aqui o professor gera questões CONTEXTUALIZADAS no
estilo SAEB para qualquer habilidade do banco — inclusive as que não são só
cálculo. O preço é que o professor também erra; por isso NADA entra no treino
sem passar pelo filtro: schema completo, 5 alternativas distintas, sem menção
a figura, consistência resposta_correta<->conta não reprovada e sem duplicata.

Requer HF_TOKEN no .env com acesso a Inference Providers (billing do HF).
O professor padrão pode ser trocado com --teacher (qualquer modelo de chat
disponível nos providers).

Uso:
    python src/distill_teacher.py --dry-run                # mostra o plano, não chama API
    python src/distill_teacher.py --limit-tuples 5         # teste barato (5 tuplas x 3 questões)
    python src/distill_teacher.py                          # gera para todas as tuplas do banco
    python src/distill_teacher.py --merge                  # mescla data/distill.jsonl no train.jsonl
    python src/distill_teacher.py --plano data/plano_destilacao.json --dry-run

DIVERSIDADE (fase 4): cada chamada ao professor recebe um SLOT planejado por
diversidade.planejar_lote — subtema matemático + tipo de raciocínio + contexto —
em vez de só um tema narrativo sorteado. O planejador faz round-robin sobre os
subtemas da taxonomia (data/taxonomia_subtemas.json), priorizando os menos
presentes no train.jsonl, então a cobertura é garantida por construção. Depois
da geração, além dos filtros antigos, a questão é rejeitada se:
  - "subtema_divergente": o classificador casa OUTRO subtema e nenhuma
    palavra-chave do subtema-alvo (só então consideramos o classificador
    confiante; sem casamento algum, 'outros', a questão é mantida);
  - "near_duplicata": diversidade.e_near_duplicata contra as já aceitas e
    contra os exemplos do train.jsonl da mesma (ano, habilidade) — trocar
    só números conta como duplicata.
O exemplo de treino grava USER_TEMPLATE + diversidade.sufixo_prompt(slot)
(formato único, igual ao da inferência) e meta com subtema/tipo_raciocinio/
contexto. A instrução de aderência à habilidade vai SÓ para o professor.

Formato de --plano (JSON), produzido por src/curar_diversidade.py ou à mão:
    {"itens": [{"ano": "9º", "habilidade": "H17", "descricao": "...",
                "dificuldade": "Moderado", "quantidade": 3,   # dificuldade opcional
                "subtema": "triangulos",          # opcional: fixa o subtema
                "tipo_raciocinio": "classificar"  # opcional: fixa o raciocínio
               }, ...]}
"descricao" é opcional (se faltar, usa a do banco para (ano, habilidade));
"quantidade" padrão = --per-tuple. Também aceita o formato de
src/curar_diversidade.py: "tipos_raciocinio": [ids] (o slot gira só entre eles)
e sem "dificuldade" (gira pelas dificuldades do banco para a (ano, habilidade)). Sem --plano, o plano é derivado de todas
as tuplas (ano, habilidade, descrição, dificuldade) do banco.

O arquivo data/distill.jsonl é APPEND-ONLY (seguro interromper e retomar);
duplicatas são descartadas na geração e na mesclagem. A mesclagem remove o
lote destilado anterior do train.jsonl antes de inserir o novo (idempotente,
igual ao generate_synthetic.py). data/val.jsonl nunca é tocado.
"""

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import diversidade  # noqa: E402
from extract_data import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from schema_utils import (  # noqa: E402
    IMAGE_PATTERN,
    check_consistency,
    check_structure,
    extract_questao,
    parse_json,
)

DB_PATH = ROOT / "DB" / "questoes.db"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
DISTILL_PATH = ROOT / "data" / "distill.jsonl"
PLANO_PATH = ROOT / "data" / "plano_destilacao.json"

# Qualquer modelo de chat forte disponível nos HF Inference Providers serve.
DEFAULT_TEACHER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_PER_TUPLE = 3
MAX_ANSWER_CHARS = 900  # mantém a resposta dentro do max_seq_length=1024 do treino

TEACHER_ADDENDUM = (
    " Regras adicionais para esta tarefa: a questão deve ser INÉDITA e "
    "contextualizada (situação do cotidiano brasileiro); os números devem ser "
    "escolhidos para que a conta seja exata; resolucao_passo_a_passo deve "
    "mostrar a conta completa e o resultado deve bater EXATAMENTE com o valor "
    "da alternativa de resposta_correta; cada distrator deve vir de um erro "
    "específico que um aluno cometeria; escreva o JSON em uma única linha."
)

# Mantida por compatibilidade; o eixo de contexto agora vem de diversidade.CONTEXTOS.
TEMAS = [
    "feira livre", "campeonato de futebol", "receita de bolo", "viagem de ônibus",
    "biblioteca da escola", "horta comunitária", "festa junina", "mesada e cofrinho",
    "loja de brinquedos", "campanha de reciclagem", "clube de leitura", "padaria",
    "passeio ao parque", "coleção de figurinhas", "mercado do bairro", "gincana escolar",
]


def normalizar(texto):
    """Chave de deduplicação: minúsculas, sem acentos, só alfanumérico."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = texto.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", texto.lower())


def load_tuples(limit=None):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT DISTINCT ano, habilidade, descricao_item, grau_resolucao FROM itens "
        "WHERE disciplina='Matemática' AND habilidade IS NOT NULL AND habilidade != '' "
        "AND descricao_item IS NOT NULL AND grau_resolucao IS NOT NULL "
        "ORDER BY ano, habilidade, grau_resolucao"
    ).fetchall()
    con.close()
    rows = [r for r in rows if r[0] and str(r[0]).lower() != "nan"]
    return rows[:limit] if limit else rows


ADERENCIA_ADDENDUM = (
    " A questão deve avaliar EXATAMENTE a habilidade {habilidade} do {ano} ano "
    "descrita acima e tratar do subtema matemático indicado ({subtema}); o "
    "contexto é só o cenário da situação, não muda o conteúdo matemático. Não "
    "troque de habilidade nem de subtema."
)


# --------------------------------------------------------------------------
# Plano de cobertura (subtema x raciocínio x contexto)
# --------------------------------------------------------------------------

def _rotulo_subtema(hab_tax, sid):
    for s in (hab_tax or {}).get("subtemas", []):
        if s["id"] == sid:
            return s["rotulo"]
    return sid


def _rotulo_tipo(tid):
    for t in diversidade.TIPOS_RACIOCINIO:
        if t["id"] == tid:
            return t["rotulo"]
    return tid


def carregar_plano(path, per_tuple=DEFAULT_PER_TUPLE, tuples=None):
    """Lê o plano JSON (formato documentado no topo) e devolve itens normalizados
    {ano, habilidade, descricao, dificuldade, quantidade, subtema?, tipo_raciocinio?}.
    Descrição ausente é preenchida pela primeira tupla do banco com o mesmo
    (ano, habilidade) — a chave é sempre (ano, habilidade), nunca só o código H."""
    dados = json.loads(Path(path).read_text(encoding="utf-8"))
    itens = dados["itens"] if isinstance(dados, dict) else dados
    descr, difs = {}, {}
    for ano, hab, d, g in (tuples if tuples is not None else []):
        descr.setdefault((ano, hab), d)
        if g not in difs.setdefault((ano, hab), []):
            difs[(ano, hab)].append(g)
    out = []
    for it in itens:
        ano, hab = it["ano"], it["habilidade"]
        descricao = it.get("descricao") or it.get("descricao_item") or descr.get((ano, hab))
        if not descricao:
            if tuples is None:
                descr.update({(a, h): d for a, h, d, _ in reversed(load_tuples())})
                descricao = descr.get((ano, hab))
            if not descricao:
                raise ValueError(f"item do plano sem descrição e fora do banco: {ano} {hab}")
        out.append({
            "ano": ano, "habilidade": hab, "descricao": descricao,
            "dificuldade": it.get("dificuldade") or it.get("grau_resolucao"),
            "quantidade": int(it.get("quantidade", per_tuple)),
            "subtema": it.get("subtema"), "tipo_raciocinio": it.get("tipo_raciocinio"),
            # formato do curar_diversidade: lista de raciocínios compatíveis e
            # sem dificuldade — giramos pelas dificuldades do banco da (ano, hab)
            "tipos_raciocinio": it.get("tipos_raciocinio") or [],
            "dificuldades": difs.get((ano, hab)) or ["Moderado"],
        })
    return out


def itens_de_tuplas(tuples, per_tuple):
    return [{"ano": a, "habilidade": h, "descricao": d, "dificuldade": g,
             "quantidade": per_tuple, "subtema": None, "tipo_raciocinio": None}
            for a, h, d, g in tuples]


def historico_treino(exemplos):
    """Agrupa as questões já existentes (train/distill) por (ano, habilidade).
    Serve de histórico ao planejador (subtemas menos cobertos primeiro) e de
    base para a deduplicação aproximada."""
    por_chave = {}
    for ex in exemplos:
        meta = ex.get("meta", {})
        try:
            obj = json.loads(ex["messages"][2]["content"])
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        for q in (obj or {}).get("questoes", []) if isinstance(obj, dict) else []:
            if isinstance(q, dict):
                por_chave.setdefault((meta.get("ano"), meta.get("habilidade")), []).append(q)
    return por_chave


def planejar_itens(itens, seed=42, historico=None, taxonomia=None):
    """Expande cada item do plano em slots (diversidade.planejar_lote), aplicando
    subtema/tipo fixados pelo item. Devolve lista de (item, slot). Itens cuja
    (ano, habilidade) não está na taxonomia recebem slot None (prompt sem sufixo)."""
    tax = taxonomia or diversidade.carregar_taxonomia()
    historico = historico or {}
    plano = []
    for it in itens:
        ano, hab = it["ano"], it["habilidade"]
        hab_tax = diversidade.obter_habilidade(ano, hab, tax)
        if hab_tax is None:
            plano.extend((it, None) for _ in range(it["quantidade"]))
            continue
        hist = historico.get((ano, hab), [])
        # entradas já classificadas (slots de itens anteriores) passam direto
        clas = [q if "subtema" in q else diversidade.classificar_questao(q, ano, hab, tax)
                for q in hist]
        slots = diversidade.planejar_lote(ano, hab, it["quantidade"], dificuldade=it["dificuldade"],
                                          seed=seed, historico=clas, taxonomia=tax)
        tipos_lista = it.get("tipos_raciocinio") or []
        for i, slot in enumerate(slots):
            item = it
            if not it.get("dificuldade"):
                difs = it.get("dificuldades") or ["Moderado"]
                item = dict(it, dificuldade=difs[i % len(difs)])
                slot["dificuldade"] = item["dificuldade"]
            if tipos_lista and not it.get("tipo_raciocinio") and slot["tipo_raciocinio"] not in tipos_lista:
                tipo = tipos_lista[i % len(tipos_lista)]
                slot["tipo_raciocinio"] = tipo
                slot["tipo_raciocinio_rotulo"] = _rotulo_tipo(tipo)
            if it.get("subtema"):
                slot["subtema"] = it["subtema"]
                slot["subtema_rotulo"] = _rotulo_subtema(hab_tax, it["subtema"])
            if it.get("tipo_raciocinio"):
                slot["tipo_raciocinio"] = it["tipo_raciocinio"]
                slot["tipo_raciocinio_rotulo"] = _rotulo_tipo(it["tipo_raciocinio"])
            plano.append((item, slot))
        # o histórico cresce com o plano: itens repetidos da mesma habilidade
        # (dificuldades diferentes) continuam a rotação em vez de recomeçar
        historico[(ano, hab)] = clas + [{"subtema": s["subtema"], "contexto": s["contexto"]}
                                        for s in slots]
    return plano


def resumo_cobertura(plano):
    """{(ano, hab): {"n": total, "subtemas": Counter, "tipos": Counter, "k": K}}"""
    tax = diversidade.carregar_taxonomia()
    res = {}
    for it, slot in plano:
        chave = (it["ano"], it["habilidade"])
        r = res.setdefault(chave, {"n": 0, "subtemas": Counter(), "tipos": Counter(),
                                   "contextos": Counter(), "k": 0})
        r["n"] += 1
        if slot:
            r["subtemas"][slot["subtema"]] += 1
            r["tipos"][slot["tipo_raciocinio"]] += 1
            r["contextos"][slot["contexto"]] += 1
            h = diversidade.obter_habilidade(*chave, tax)
            r["k"] = len(h["subtemas"]) if h else 0
    return res


def prompt_usuario(ano, habilidade, descricao, dificuldade, slot):
    """Prompt de TREINO: USER_TEMPLATE + sufixo único de diversidade."""
    user = USER_TEMPLATE.format(quantidade=1, ano=ano, habilidade=habilidade,
                                descricao=descricao, dificuldade=dificuldade)
    return user + (diversidade.sufixo_prompt(slot) if slot else "")


def prompt_professor(ano, habilidade, descricao, dificuldade, slot):
    """Prompt do PROFESSOR = prompt de treino + instrução explícita de aderência."""
    user = prompt_usuario(ano, habilidade, descricao, dificuldade, slot)
    if slot:
        user += ADERENCIA_ADDENDUM.format(habilidade=habilidade, ano=ano,
                                          subtema=slot["subtema_rotulo"])
    return user


def subtema_divergente(questao, ano, habilidade, slot, taxonomia=None):
    """True só quando o classificador é CONFIANTE de que a questão é de outro
    subtema: casou um subtema diferente do alvo e zero palavras-chave do alvo.
    Casamento nenhum ('outros') ou K=1 não rejeita — o léxico é ruidoso e a
    ausência de evidência não é evidência contrária."""
    if not slot:
        return False
    tax = taxonomia or diversidade.carregar_taxonomia()
    hab = diversidade.obter_habilidade(ano, habilidade, tax)
    if not hab or len(hab["subtemas"]) <= 1:
        return False
    classe = diversidade.classificar_questao(questao, ano, habilidade, tax)["subtema"]
    if classe in (slot["subtema"], "outros"):
        return False
    alvo = next((s for s in hab["subtemas"] if s["id"] == slot["subtema"]), None)
    if alvo is None:
        return False
    texto = diversidade.normalizar_texto(diversidade.texto_questao(questao))
    regs = alvo["palavras_chave"] if isinstance(alvo["palavras_chave"], list) else [alvo["palavras_chave"]]
    return sum(len(re.findall(r, texto)) for r in regs if r) == 0


def filtrar(obj, raw_text, vistos, strict=False, slot=None, ano=None, habilidade=None,
            aceitas=None, taxonomia=None):
    """Aplica os filtros determinísticos. Retorna None se aprovado, ou o motivo.

    `obj` é o wrapper `{"questoes": [...]}` bruto devolvido pelo professor;
    só a primeira questão é considerada (o pedido é sempre quantidade=1).
    Ordem: estrutura, figura, tamanho, consistência, duplicata exata e, com
    slot/aceitas, subtema_divergente e near_duplicata (filtros de diversidade
    vêm por último para não mascarar as rejeições de qualidade)."""
    if obj is None:
        return "json_invalido"
    flags = check_structure(obj, quantidade_esperada=1)
    if not (flags["wrapper_valido"] and flags["schema_completo"] and flags["resposta_valida"]
            and flags["alternativas_distintas"] and flags["difficulty_valida"]):
        return "estrutura"
    questao = extract_questao(obj, 0)
    blob = json.dumps(questao, ensure_ascii=False)
    if IMAGE_PATTERN.search(blob):
        return "menciona_figura"
    if len(blob) > MAX_ANSWER_CHARS:
        return "muito_longa"
    consistente, _ = check_consistency(questao)
    if consistente is False:
        return "resposta_inconsistente"
    if strict and consistente is not True:
        return "consistencia_nao_verificavel"
    chave = normalizar(questao.get("enunciado", ""))
    if not chave or chave in vistos:
        return "duplicata"
    if ano is not None and subtema_divergente(questao, ano, habilidade, slot, taxonomia):
        return "subtema_divergente"
    if aceitas is not None:
        # aceitas: dict (ano, hab) -> questões (treino + já aceitas). A mesma
        # chave (ano, habilidade) — H17 do 9º não compete com H17 do 5º.
        for outra in aceitas.get((ano, habilidade), []):
            if diversidade.e_near_duplicata(questao, outra):
                return "near_duplicata"
        aceitas.setdefault((ano, habilidade), []).append(questao)
    vistos.add(chave)
    return None


def build_example(ano, habilidade, descricao, dificuldade, questao, idx, teacher, slot=None):
    ex = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt_usuario(ano, habilidade, descricao, dificuldade, slot),
            },
            {"role": "assistant", "content": json.dumps({"questoes": [questao]}, ensure_ascii=False)},
        ],
        "meta": {
            "codigo_item": f"DIST-{habilidade}-{dificuldade}-{idx:05d}",
            "ano": ano,
            "habilidade": habilidade,
            "dificuldade": dificuldade,
            "destilado": True,
            "professor": teacher,
        },
    }
    if slot:
        ex["meta"].update({"subtema": slot["subtema"], "tipo_raciocinio": slot["tipo_raciocinio"],
                           "contexto": slot["contexto"]})
    return ex


def montar_plano(args):
    """Plano (item, slot) a partir de --plano ou de todas as tuplas do banco,
    com o train.jsonl + distill.jsonl como histórico de cobertura."""
    tuples = load_tuples()
    if args.plano:
        itens = carregar_plano(args.plano, args.per_tuple, tuples)
    else:
        itens = itens_de_tuplas(tuples, args.per_tuple)
    if args.limit_tuples:
        itens = itens[:args.limit_tuples]
    exemplos = []
    for p in (TRAIN_PATH, DISTILL_PATH):
        if p.exists():
            exemplos += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    hist = historico_treino(exemplos)
    plano = planejar_itens(itens, seed=args.seed, historico={k: list(v) for k, v in hist.items()})
    return itens, plano, hist


def gerar(args, client=None):
    """Gera com o professor seguindo o plano. `client` permite injetar um
    cliente falso nos testes (precisa só de chat_completion)."""
    if client is None:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=args.teacher, provider=args.provider)
    itens, plano, hist = montar_plano(args)

    vistos = set()
    aceitos_previos = 0
    if DISTILL_PATH.exists():
        for line in open(DISTILL_PATH, encoding="utf-8"):
            ex = json.loads(line)
            obj = json.loads(ex["messages"][2]["content"])
            questao = extract_questao(obj, 0)
            vistos.add(normalizar((questao or {}).get("enunciado", "")))
            aceitos_previos += 1
        print(f"Retomando: {aceitos_previos} questões já aceitas em {DISTILL_PATH} "
              "(duplicatas serão descartadas).")
    # base da dedup aproximada: treino real/sintético + destilado já aceito
    aceitas = {k: list(v) for k, v in hist.items()}

    rejeicoes = Counter()
    aceitos, erros_api, idx = 0, 0, aceitos_previos
    print(f"Professor: {args.teacher} (provider: {args.provider})")
    print(f"{len(itens)} itens de plano -> {len(plano)} chamadas planejadas\n")

    with open(DISTILL_PATH, "a", encoding="utf-8") as saida:
        for c, (it, slot) in enumerate(plano, 1):
            ano, habilidade = it["ano"], it["habilidade"]
            descricao, dificuldade = it["descricao"], it["dificuldade"]
            user = prompt_professor(ano, habilidade, descricao, dificuldade, slot)
            try:
                resp = client.chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT + TEACHER_ADDENDUM},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:  # rede/quota/provider — segue para a próxima
                erros_api += 1
                print(f"  [erro API {erros_api}/{args.max_errors}] {exc}")
                if erros_api >= args.max_errors:
                    raise SystemExit(
                        "Muitos erros de API — abortando. O que já foi aceito "
                        f"está salvo em {DISTILL_PATH}; rode de novo para retomar."
                    )
                time.sleep(args.retry_wait)
                continue

            obj = parse_json(text)
            motivo = filtrar(obj, text, vistos, strict=args.strict, slot=slot, ano=ano,
                             habilidade=habilidade, aceitas=aceitas)
            if motivo:
                rejeicoes[motivo] += 1
                continue

            idx += 1
            questao = extract_questao(obj, 0)
            ex = build_example(ano, habilidade, descricao, dificuldade, questao, idx,
                               args.teacher, slot)
            saida.write(json.dumps(ex, ensure_ascii=False) + "\n")
            saida.flush()
            aceitos += 1
            if c % 10 == 0 or c == len(plano):
                print(f"[{c}/{len(plano)}] {aceitos} aceitas até agora")

    print(f"\nAceitas nesta rodada: {aceitos} (total no arquivo: {aceitos_previos + aceitos})")
    print(f"Rejeitadas por motivo: {dict(rejeicoes) or 'nenhuma'}")
    print(f"Erros de API: {erros_api}")
    print(f"\nPróximo passo: revisar amostras de {DISTILL_PATH} e mesclar com:")
    print("    python src/distill_teacher.py --merge")
    return {"aceitos": aceitos, "rejeicoes": dict(rejeicoes), "erros_api": erros_api}


def dry_run(args):
    """Mostra o plano de cobertura por (ano, habilidade) sem chamar a API."""
    itens, plano, _ = montar_plano(args)
    print(f"Professor: {args.teacher} (provider: {args.provider})")
    print(f"Itens de plano: {len(itens)} | chamadas planejadas: {len(plano)}"
          + (f" | plano: {args.plano}" if args.plano else " | plano: todas as tuplas do banco"))
    print("Filtros: schema completo + 5 alternativas distintas + sem figura + "
          "consistência não reprovada + sem duplicata + subtema aderente + sem near-duplicata"
          + (" + consistência VERIFICADA (--strict)" if args.strict else ""))
    print("\nCobertura planejada (subtemas distintos / K da taxonomia):")
    for (ano, hab), r in resumo_cobertura(plano).items():
        subs = ", ".join(f"{s}={n}" for s, n in r["subtemas"].most_common())
        print(f"  {ano} {hab}: {r['n']} chamadas | {len(r['subtemas'])}/{r['k']} subtemas "
              f"| {len(r['tipos'])} raciocínios | {len(r['contextos'])} contextos | {subs}")
    if plano:
        it, slot = plano[0]
        print("\nExemplo de prompt que será enviado ao professor:")
        print(f"  [system] {SYSTEM_PROMPT[:120]}... (+ addendum de destilação)")
        print("  [user]   " + prompt_professor(it["ano"], it["habilidade"], it["descricao"],
                                              it["dificuldade"], slot))
        print("  [treino] " + prompt_usuario(it["ano"], it["habilidade"], it["descricao"],
                                            it["dificuldade"], slot))
    return plano


def merge():
    if not DISTILL_PATH.exists():
        raise SystemExit(f"{DISTILL_PATH} não existe — rode a geração primeiro.")
    destilados = [json.loads(l) for l in open(DISTILL_PATH, encoding="utf-8")]

    # Dedup interno (retomadas podem ter re-aceitado enunciados parecidos).
    vistos, unicos = set(), []
    for ex in destilados:
        obj = json.loads(ex["messages"][2]["content"])
        questao = extract_questao(obj, 0)
        chave = normalizar((questao or {}).get("enunciado", ""))
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(ex)

    base = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")] if TRAIN_PATH.exists() else []
    base_sem_destilado = [ex for ex in base if not ex.get("meta", {}).get("destilado")]
    combinado = base_sem_destilado + unicos
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in combinado:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_real = sum(1 for ex in base_sem_destilado if not ex.get("meta", {}).get("sintetico"))
    n_sint = len(base_sem_destilado) - n_real
    print(f"Treino real:      {n_real}")
    print(f"Treino sintético: {n_sint}")
    print(f"Treino destilado: {len(unicos)} ({len(destilados) - len(unicos)} duplicatas removidas)")
    print(f"Total escrito em {TRAIN_PATH}: {len(combinado)}")
    print("\ndata/val.jsonl não foi tocado — validação continua 100% com questões reais.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--teacher", default=DEFAULT_TEACHER,
                        help=f"modelo professor (padrão: {DEFAULT_TEACHER})")
    parser.add_argument("--provider", default="auto",
                        help="HF Inference Provider (padrão: auto)")
    parser.add_argument("--per-tuple", type=int, default=DEFAULT_PER_TUPLE,
                        help="questões pedidas por (ano, habilidade, descrição, dificuldade)")
    parser.add_argument("--limit-tuples", type=int, default=None,
                        help="limita o nº de tuplas (para teste barato)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="temperatura do professor (alta = mais diversidade)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--strict", action="store_true",
                        help="exige consistência VERIFICADA (descarta as não verificáveis)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--merge", action="store_true",
                        help="mescla data/distill.jsonl no data/train.jsonl e sai")
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra o plano de cobertura sem chamar a API")
    parser.add_argument("--plano", default=None,
                        help="JSON de plano (ex.: data/plano_destilacao.json); formato no topo")
    args = parser.parse_args()

    if args.merge:
        merge()
        return

    if args.dry_run:
        dry_run(args)
        return

    gerar(args)


if __name__ == "__main__":
    main()
