"""Geração PLANEJADA de um lote de N questões (inferência offline, GGUF).

Por que existe: pedir "Gere N questões" numa única chamada deixava a
diversidade por conta da amostragem — e só a 1ª questão do lote passava por
check_consistency/fix_gabarito/depende_de_visual_ausente. No teste real, 9º H17
saiu 3/3 em triângulos. Aqui cada questão é gerada individualmente, guiada por
um plano de cobertura de subtemas (diversidade.planejar_lote), e TODAS passam
pelo mesmo pipeline validado de produção (test_model.generate_validated).

Fluxo, para cada slot do plano:
  1. prompt = USER_TEMPLATE(quantidade=1) + diversidade.sufixo_prompt(slot)
     (+ restrição explícita na regeneração). A gramática GBNF é mantida.
  2. generate_validated: estrutura, consistência, fix_gabarito, visual ausente.
  3. classificar + violacoes_diversidade contra o lote já aceito.
  4. Se violar: regenera com diversidade.montar_restricao e seed diferente,
     até `max_tentativas_diversidade` vezes.
  5. Se esgotar: fica o MELHOR candidato, com QUALIDADE PRIMEIRO — uma questão
     válida nunca é trocada por uma inválida só por ser mais diversa. A
     violação remanescente é registrada.
Ao fim monta o wrapper {"questoes": [...]} (schema inalterado) e calcula as
métricas de diversidade do lote.

Custo: cada tentativa é uma chamada ao llama-cli (~10-15 s em CPU com 4
threads, recarregando o modelo). Um lote de N custa no mínimo N chamadas; as
regenerações de diversidade somam no pior caso N*max_tentativas chamadas.

Uso típico (pela CLI de test_model.py, onde é o padrão para --quantidade > 1):
    python src/test_model.py --ano "9º" --habilidade H17 --quantidade 5
"""
import random
import time

import diversidade as dv
from extract_data import USER_TEMPLATE
from schema_utils import check_structure
from test_model import MAX_NEW_TOKENS, generate_validated

# Qualidade do pós-processamento, do melhor para o pior. É o critério
# PRIMÁRIO na escolha entre candidatos de um slot; diversidade só desempata.
RANK_STATUS = {"ok": 4, "nao_verificavel": 4, "corrigido": 3,
               "depende_de_visual": 1, "falha": 0}


def _slots_fallback(quantidade, dificuldade):
    """Plano mínimo quando (ano, habilidade) não está na taxonomia: sem subtema
    (não inventamos um), mas o contexto ainda gira para não repetir cenário."""
    return [{"indice": i, "subtema": "geral", "subtema_rotulo": "geral",
             "tipo_raciocinio": "indefinido", "tipo_raciocinio_rotulo": "livre",
             "contexto": dv.CONTEXTOS[i % len(dv.CONTEXTOS)]["id"],
             "contexto_rotulo": dv.CONTEXTOS[i % len(dv.CONTEXTOS)]["rotulo"],
             "estrutura": "pergunta_direta", "estrutura_rotulo": "pergunta direta",
             "dificuldade": dificuldade} for i in range(quantidade)]


def montar_prompt(ano, habilidade, descricao, dificuldade, slot, restricao=None):
    """USER_TEMPLATE de produção (quantidade=1) + sufixo único de diversidade.
    O sufixo é o mesmo texto usado na destilação/treino (paridade)."""
    p = USER_TEMPLATE.format(quantidade=1, ano=ano, habilidade=habilidade,
                             descricao=descricao, dificuldade=dificuldade)
    if slot.get("subtema") != "geral" or slot.get("tipo_raciocinio") != "indefinido":
        p += dv.sufixo_prompt(slot)
    else:
        p += f" Contexto: {slot['contexto_rotulo']}."
    if restricao:
        p += f" Restrição: {restricao}"
    return p


def _chave(cand):
    """Ordenação de candidatos: qualidade primeiro, depois menos violações."""
    return (RANK_STATUS.get(cand["status"], 0) if cand["obj"] is not None else -1,
            -len(cand["violacoes"]))


def gerar_lote_planejado(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                         quantidade, threads, grammar=None, retries=1, base_seed=None,
                         max_tentativas_diversidade=2, gen_fn=None, historico=None,
                         taxonomia=None, verbose=False):
    """Gera `quantidade` questões com plano de subtemas e regeneração guiada.

    gen_fn: substituto de test_model.generate (mesma assinatura) — para testes
    sem llama-cli. base_seed: torna o lote reprodutível (plano + seeds); None
    sorteia uma base por lote, de modo que tentativas/slots nunca repetem seed
    (no modo antigo toda regeneração usava a seed fixa 1001).

    Retorna dict: obj (wrapper {"questoes": [...]}), questoes, plano, metricas,
    violacoes (por slot, as que restaram), tempo_s, regeneracoes_diversidade,
    flags (check_structure do wrapper), detalhes (por slot).
    """
    t0 = time.perf_counter()
    quantidade = int(quantidade)
    if base_seed is None:
        base_seed = random.randrange(1, 10 ** 6)
    try:
        plano = dv.planejar_lote(ano, habilidade, quantidade, dificuldade,
                                 seed=base_seed, historico=historico, taxonomia=taxonomia)
        tem_taxonomia = True
    except (KeyError, FileNotFoundError):
        plano, tem_taxonomia = _slots_fallback(quantidade, dificuldade), False

    aceitas, detalhes, violacoes_final = [], [], []
    regen_div = 0
    for slot in plano:
        t_slot = time.perf_counter()
        melhor, restricao, candidatos = None, None, []
        for tent in range(max_tentativas_diversidade + 1):
            prompt = montar_prompt(ano, habilidade, descricao, dificuldade, slot, restricao)
            # Seed distinta por (lote, slot, tentativa de diversidade);
            # generate_validated ainda soma o índice da tentativa de qualidade.
            seed_item = base_seed * 1000 + slot["indice"] * 10 + tent
            r = generate_validated(llama_cli, gguf_path, prompt, threads, MAX_NEW_TOKENS,
                                   grammar=grammar, retries=retries, quantidade=1,
                                   base_seed=seed_item, gen_fn=gen_fn)
            viol = []
            if r["obj"] is not None and tem_taxonomia:
                viol = dv.violacoes_diversidade(aceitas, r["obj"], slot, ano, habilidade,
                                                quantidade, taxonomia=taxonomia)
            cand = {"obj": r["obj"], "status": r["status"], "violacoes": viol,
                    "regeneracoes": r["regeneracoes"], "elapsed": r["elapsed"],
                    "gen_tps": r["gen_tps"], "tentativa": tent}
            candidatos.append(cand)
            if melhor is None or _chave(cand) > _chave(melhor):
                melhor = cand
            # Para quando o candidato é bom E diverso. Se é diverso mas de
            # qualidade inferior, generate_validated já esgotou os retries dele;
            # uma nova tentativa aqui só gastaria tempo sem motivo de diversidade.
            # Candidato estruturalmente quebrado/sem questão ("falha") usa as
            # tentativas restantes como novas chances de qualidade (seed nova),
            # mesmo sem violação de diversidade.
            if not viol and RANK_STATUS.get(cand["status"], 0) > 0:
                break
            if not viol:
                continue
            if tent < max_tentativas_diversidade:
                regen_div += 1
                restricao = dv.montar_restricao(viol, habilidade, slot)
                if verbose:
                    print(f"  [slot {slot['indice']}] regenerando: {restricao}")

        if melhor["obj"] is not None:
            aceitas.append(melhor["obj"])
        violacoes_final.append({"indice": slot["indice"],
                                "violacoes": melhor["violacoes"]})
        detalhes.append({
            "indice": slot["indice"], "subtema_planejado": slot["subtema"],
            "classificacao": (dv.classificar_questao(melhor["obj"], ano, habilidade, taxonomia)
                              if melhor["obj"] is not None and tem_taxonomia else None),
            "status": melhor["status"], "tentativas_diversidade": len(candidatos),
            "tentativa_escolhida": melhor["tentativa"],
            "regeneracoes_qualidade": sum(c["regeneracoes"] for c in candidatos),
            "tempo_modelo_s": round(sum(c["elapsed"] for c in candidatos), 2),
            "tempo_s": round(time.perf_counter() - t_slot, 2),
            "gen_tps": melhor["gen_tps"],
        })

    wrapper = {"questoes": aceitas}
    metricas = None
    if tem_taxonomia and aceitas:
        metricas = dv.diversity_score(aceitas, ano, habilidade, taxonomia=taxonomia)
        metricas.pop("classificacoes", None)  # já estão em detalhes
    tempo = time.perf_counter() - t0
    return {
        "obj": wrapper, "questoes": aceitas, "plano": plano, "metricas": metricas,
        "violacoes": violacoes_final, "tempo_s": round(tempo, 2),
        "tempo_medio_por_questao_s": round(tempo / max(1, quantidade), 2),
        "regeneracoes_diversidade": regen_div,
        "flags": check_structure(wrapper, quantidade_esperada=quantidade),
        "detalhes": detalhes, "base_seed": base_seed, "planejado": tem_taxonomia,
    }
