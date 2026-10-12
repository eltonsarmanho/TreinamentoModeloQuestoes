"""Plano da injeção direcionada do ciclo v6 (etapa 6). Sem API, sem escrever na base.

Reutiliza o que já existe — não reimplementa:
  - injetar_questoes.lacunas()        : déficit por habilidade a partir de relatorio_base.py
                                        (meta >= 30, >= 3 por subtema, dificuldades equilibradas);
  - injetar_questoes.planejar_slots() : dificuldade + slot de diversidade (subtema, tipo de
                                        raciocínio, contexto, estrutura) via diversidade.planejar_lote,
                                        começando pelos subtemas menos cobertos da base;
  - perfil_base.py                    : subtemas vazios, concentração, entropia de raciocínio.

Acrescenta, por slot: representação, restrições específicas e o NÍVEL DE EVIDÊNCIA exigido para
aceitar o item na candidata — e ordena as habilidades por uma PRIORIDADE que não é a meta
uniforme: pesa lacuna curricular (subtemas vazios da taxonomia), fração da meta faltando,
concentração num subtema e pouca variedade de raciocínio.

Contexto narrativo NÃO conta como diversidade: dois slots com o mesmo subtema e o mesmo tipo de
raciocínio são a mesma questão matemática, mesmo com contextos diferentes. O plano só repete
(subtema, tipo) quando a habilidade não tem mais combinações; isso é medido e reportado.

    venv/bin/python src/planejar_injecao_v6.py --relatorio outputs/melhoria_v6/relatorio_base_v5.json \
        --perfil outputs/melhoria_v6/perfil_v5.json --base data/train_curado_v5.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import diversidade as dv  # noqa: E402
from perfil_base import SAIDA_DIR, carregar  # noqa: E402

SEED = 20261010
PESOS_PRIORIDADE = {"subtemas_vazios": 3.0, "fracao_deficit": 2.0, "concentracao": 1.0, "raciocinio": 1.0}
NIVEL_BASE = ["estrutura_json", "sem_dependencia_visual", "verificador_aritmetico_nao_reprova",
              "classificacao_habilidade_subtema", "sem_duplicata_nem_near_dup", "sem_contaminacao_val",
              "validador_cego", "revisor_cego"]


def prioridade(hab_rel, hab_perfil):
    """(score, componentes). Todos os componentes em [0, 1]."""
    k = max(1, hab_perfil.get("subtemas_taxonomia", 1)) if hab_perfil else 1
    vazios = len(hab_perfil.get("subtemas_vazios", [])) / k if hab_perfil else 0.0
    frac = min(1.0, hab_rel["deficit"] / max(1, hab_rel["meta"]))
    conc = max(0.0, (hab_perfil.get("subtema_dominante_pct", 0) - 60) / 40) if hab_perfil and k > 1 else 0.0
    ent = (hab_perfil or {}).get("diversidade", {}).get("entropia_raciocinio")
    rac = 0.0 if ent is None else max(0.0, 1 - ent)
    # Classificador léxico que manda quase tudo para 'outros' não prova lacuna: "subtema vazio" e
    # "concentração" valem na proporção do que ele conseguiu classificar (5º H21: 100% 'outros').
    outros = (hab_perfil or {}).get("diversidade", {}).get("pct_outros", 0) / 100
    vazios, conc = vazios * (1 - outros), conc * (1 - outros)
    comp = {"subtemas_vazios": round(vazios, 3), "fracao_deficit": round(frac, 3), "concentracao": round(conc, 3),
            "raciocinio": round(rac, 3), "verificar_classificador": outros >= 0.5}
    return round(sum(PESOS_PRIORIDADE[c] * comp[c] for c in PESOS_PRIORIDADE), 3), comp


def representacao_para(slot, hab_tax, perfil_rep):
    sub = slot.get("subtema") or ""
    if dv.exige_dados_textuais(sub):
        return "tabela_textual"
    # a menos usada na habilidade entre as que fazem sentido para o subtema
    opcoes = ["numerica", "verbal", "medidas"]
    return min(opcoes, key=lambda r: (perfil_rep.get(r, 0), opcoes.index(r)))


def restricoes_para(slot, hab_tax):
    r = ["não use 'Nenhuma das alternativas anteriores': as 5 alternativas são respostas reais",
         "cada alternativa errada é o resultado exato de um erro nomeado de aluno deste ano",
         "exatamente uma alternativa correta; alternativas no mesmo espaço de resposta e na mesma unidade",
         "resolução passo a passo didática, com as contas explícitas e sem citar letras de alternativas"]
    if dv.exige_dados_textuais(slot.get("subtema")):
        r.append("todos os dados da tabela/gráfico escritos no enunciado ('rótulo: valor')")
    if hab_tax and hab_tax.get("conceitual"):
        r.append("habilidade conceitual: nenhuma classe mais geral pode também ser verdadeira (D5)")
    if slot.get("dificuldade") == "Difícil":
        r.append("Difícil genuíno: duas ou mais etapas, ou conceito menos direto")
    return r


def nivel_evidencia(ano, hab, slot, hab_tax):
    n = list(NIVEL_BASE)
    if hab_tax and hab_tax.get("conceitual"):
        n.append("segundo_revisor_gemini")  # outra família: o par Sabiá sozinho não basta em item conceitual
    if (ano, hab) == ("9º", "H17"):
        n.append("verificador_geometria_ok")
    if dv.exige_dados_textuais(slot.get("subtema")):
        n.append("tem_dados_textuais")
    # Dificuldade NÃO entra como evidência exigida: a D3 (dificuldade_real do revisor = pedida) está
    # suspensa desde 2026-10-01 — o revisor disse "Fácil" para 83% dos Moderado/Difícil reais do
    # banco. Ela fica registrada no meta e é conferida na amostra humana cega.
    return n


def planejar(relatorio, perfil, base_questoes, taxonomia, top=None, meta_por_hab=None):
    import injetar_questoes as iq
    lista, ignoradas = iq.lacunas(relatorio, meta_por_hab=meta_por_hab)
    hp = perfil["habilidades_detalhe"]
    for lac in lista:
        lac["prioridade"], lac["componentes"] = prioridade(lac, hp.get(f"{lac['ano']}|{lac['habilidade']}"))
    lista.sort(key=lambda x: (-x["prioridade"], x["ano"], x["habilidade"]))
    if top:
        lista = lista[:top]
    por_hab = {}
    for i, j, m, q, _p in base_questoes:
        por_hab.setdefault((m["ano"], m["habilidade"]), []).append(q)
    plano, combos_repetidas = [], 0
    for lac in lista:
        ano, hab = lac["ano"], lac["habilidade"]
        hab_tax = dv.obter_habilidade(ano, hab, taxonomia)
        hist = por_hab.get((ano, hab), [])
        rep_perfil = Counter(dv.classificar_questao(q, ano, hab, taxonomia)["representacao"] for q in hist)
        slots = iq.planejar_slots(lac, lac["alvo"], {}, hist, taxonomia, SEED)
        vistos = Counter()
        itens = []
        for d, s in slots:
            s = dict(s or {"subtema": None, "tipo_raciocinio": None, "contexto": None, "estrutura": None})
            s["dificuldade"] = d
            par = (s.get("subtema"), s.get("tipo_raciocinio"))
            vistos[par] += 1
            s["representacao"] = representacao_para(s, hab_tax, rep_perfil)
            rep_perfil[s["representacao"]] += 1
            s["restricoes"] = restricoes_para(s, hab_tax)
            s["nivel_evidencia"] = nivel_evidencia(ano, hab, s, hab_tax)
            itens.append(s)
        repetidas = sum(v - 1 for v in vistos.values() if v > 1)
        combos_repetidas += repetidas
        combos_possiveis = sum(len(st.get("tipos_raciocinio") or hab_tax["tipos_raciocinio"])
                               for st in hab_tax["subtemas"]) if hab_tax else None
        plano.append({"ano": ano, "habilidade": hab, "descricao": lac["descricao"], "prioridade": lac["prioridade"],
                      "componentes": lac["componentes"], "deficit": lac["deficit"], "alvo": lac["alvo"],
                      "dificuldades": dict(Counter(s["dificuldade"] for s in itens)),
                      "combinacoes_subtema_tipo_possiveis": combos_possiveis,
                      "slots_que_repetem_subtema_e_tipo": repetidas, "slots": itens})
    return {"seed": SEED, "pesos_prioridade": PESOS_PRIORIDADE, "habilidades": len(plano),
            "slots": sum(p["alvo"] for p in plano), "slots_que_repetem_subtema_e_tipo": combos_repetidas,
            "ignoradas": ignoradas, "plano": plano}


def custo_medido_por_aceita(logs, aceitas, precos):
    """R$ por questão aceita nas injeções anteriores: custo de TODAS as chamadas (inclusive as das
    rejeitadas) dividido pelas aceitas. Preços de ciclo_v6/precos_maritaca.json."""
    import sabia_v6 as sv
    total, n = 0.0, 0
    for f in logs:
        for linha in open(f, encoding="utf-8"):
            try:
                r = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if r.get("simulado"):
                continue
            total += sv.custo_brl(precos, r["modelo"], r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0)
            n += 1
    return {"chamadas": n, "custo_brl": round(total, 2), "aceitas": aceitas,
            "brl_por_aceita": round(total / aceitas, 4) if aceitas else None}


def estimar(n_slots, custo_por_aceita, brl_por_aceita=None, fator_margem=1.5):
    """Custo previsto a partir do MEDIDO nas injeções anteriores. fator_margem: teto prudente para
    habilidades com aceite pior que a média (9º H17 rendeu 1 aceita em 12 no piloto 2)."""
    out = {"slots": n_slots, "chamadas_por_aceita": custo_por_aceita["chamadas"],
           "tokens_por_aceita": custo_por_aceita["tokens"],
           "chamadas_previstas": int(round(n_slots * custo_por_aceita["chamadas"]))}
    if brl_por_aceita:
        out["brl_por_aceita_medido"] = brl_por_aceita
        out["custo_previsto_brl"] = round(n_slots * brl_por_aceita, 2)
        out["custo_teto_brl"] = round(n_slots * brl_por_aceita * fator_margem, 2)
        out["chamadas_teto"] = int(round(out["chamadas_previstas"] * fator_margem))
    return out


def comando(plano, saida_dir):
    habs = ",".join(f"{p['ano']}:{p['habilidade']}" for p in plano["plano"])
    return ("venv/bin/python src/injetar_questoes.py --relatorio outputs/melhoria_v6/relatorio_base_v5.json "
            "--train data/train_curado_v5.jsonl "
            f"--habilidades {habs} --saida {saida_dir}/injecao_v6.jsonl --rejeitadas {saida_dir}/injecao_v6_rejeitadas.jsonl "
            f"--resumo {saida_dir}/injecao_v6_resumo.json --log-uso {saida_dir}/uso_api_injecao_v6.jsonl "
            "--cache-juizes outputs/agentes/juizes_cache.jsonl --max-chamadas <N> --dry-run")


def resumo_md(p, est):
    L = ["# Plano de injeção direcionada — ciclo v6", "", "Gerado por `src/planejar_injecao_v6.py`, sem API. "
         "Nada foi gerado: este é o escopo proposto para autorização.", "",
         f"- Habilidades: **{p['habilidades']}**, slots: **{p['slots']}**; slots que repetem (subtema, tipo) "
         f"dentro da habilidade: {p['slots_que_repetem_subtema_e_tipo']}.",
         f"- Pesos da prioridade: {p['pesos_prioridade']}.",
         f"- Chamadas previstas (medido: {est['chamadas_por_aceita']} chamadas por questão aceita): ~{est['chamadas_previstas']}; "
         f"custo previsto ~R$ {est.get('custo_previsto_brl')} (medido R$ {est.get('brl_por_aceita_medido')} por aceita), "
         f"teto prudente ~R$ {est.get('custo_teto_brl')} / {est.get('chamadas_teto')} chamadas (margem 1,5 para habilidades de aceite baixo). "
         "Gemini (2º revisor das habilidades conceituais) não incluído.",
         "- `vazios` e `concentração` já descontam a fração que o classificador léxico deixou em 'outros'; "
         "⚠ = mais da metade em 'outros' (conferir o classificador antes de concluir que falta subtema).",
         "", "| # | ano|hab | prioridade | vazios | déficit/meta | concentração | raciocínio | alvo | F/M/D | (sub,tipo) possíveis | repetidos |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, h in enumerate(p["plano"], 1):
        c, d = h["componentes"], h["dificuldades"]
        aviso = " ⚠" if c.get("verificar_classificador") else ""
        L.append(f"| {i} | {h['ano']}|{h['habilidade']}{aviso} | {h['prioridade']} | {c['subtemas_vazios']} | {c['fracao_deficit']} | "
                 f"{c['concentracao']} | {c['raciocinio']} | {h['alvo']} | {d.get('Fácil', 0)}/{d.get('Moderado', 0)}/"
                 f"{d.get('Difícil', 0)} | {h['combinacoes_subtema_tipo_possiveis']} | {h['slots_que_repetem_subtema_e_tipo']} |")
    if p["ignoradas"]:
        L += ["", "Dificuldades que o banco real não usa (não pedidas, regra existente da injeção): "
              + "; ".join(f"{x['ano']} {x['habilidade']} {x['falta_ignorada']}" for x in p["ignoradas"])]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relatorio", default=str(SAIDA_DIR / "relatorio_base_v5.json"))
    ap.add_argument("--perfil", default=str(SAIDA_DIR / "perfil_v5.json"))
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v5.jsonl"))
    ap.add_argument("--top", type=int, default=None, help="só as K habilidades de maior prioridade")
    ap.add_argument("--meta-por-habilidade", type=int, default=None, help="teto de slots por habilidade")
    ap.add_argument("--saida-dir", default=str(SAIDA_DIR))
    args = ap.parse_args()
    rel = json.loads(Path(args.relatorio).read_text(encoding="utf-8"))
    perfil = json.loads(Path(args.perfil).read_text(encoding="utf-8"))
    p = planejar(rel, perfil, carregar(args.base), dv.carregar_taxonomia(), args.top, args.meta_por_habilidade)
    cpa = json.loads((ROOT / "outputs" / "agentes" / "injecao_resumo.json").read_text(encoding="utf-8"))["custo_por_aceita"]
    import sabia_v6 as sv
    aceitas = sum(1 for l in open(ROOT / "data" / "injecao_saeb.jsonl", encoding="utf-8") if l.strip())
    medido = custo_medido_por_aceita(sorted((ROOT / "outputs" / "agentes" / "escala").glob("uso_api_injecao_*.jsonl")),
                                     aceitas, sv.carregar_precos())
    est = estimar(p["slots"], cpa, medido["brl_por_aceita"])
    est["medicao"] = medido
    p["estimativa"] = est
    p["comando_sugerido_dry_run"] = comando(p, Path(args.saida_dir).relative_to(ROOT) if Path(args.saida_dir).is_absolute() else args.saida_dir)
    out = Path(args.saida_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plano_injecao_v6.json").write_text(json.dumps(p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "plano_injecao_v6.md").write_text(resumo_md(p, est), encoding="utf-8")
    print(f"{p['habilidades']} habilidades, {p['slots']} slots, repetidos={p['slots_que_repetem_subtema_e_tipo']}; "
          f"chamadas previstas ~{est['chamadas_previstas']}")


if __name__ == "__main__":
    main()
