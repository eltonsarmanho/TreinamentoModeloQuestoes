"""Monta a candidata data/train_curado_v6.jsonl (ciclo v6, etapa 9). Sem API; a v5 nunca é escrita.

Operações (cada uma vira uma linha em outputs/melhoria_v6/proveniencia_v6.jsonl):
  mantida             — exemplo da v5 sem alteração;
  resolucao_aprimorada — v5 + resolução reescrita APROVADA (melhorar_resolucoes: verificadores +
                         avaliador). Só o campo resolucao_passo_a_passo muda; enunciado, alternativas e
                         gabarito (por conteúdo) são os da v5. A original fica na proveniência.
  injetada            — exemplo de data/injecao_v6.jsonl que passou nos filtros e juízes da injeção E nas
                         travas desta montagem (sem duplicata exata nem near-dup contra o que já está na
                         candidata, sem contaminação com val*, verificador aritmético não reprova).
  removida_duplicata  — item NÃO real da v5 que é duplicata provável de outro (o real, ou o anterior, fica).
  quarentena          — injetada de habilidade conceitual sem 2º revisor (revisor2 != True), ou item com
                         alternativas de mesmo valor (revisão humana). Não entra na candidata.
Nenhuma questão real (real_banco, real_L2) é removida. Nenhum gabarito muda por conteúdo.
As letras do gabarito são re-balanceadas SÓ entre as injetadas (permutar_lote) se a base passar de 19-21%.

    venv/bin/python src/montar_candidata_v6.py
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agentes_questoes as aq  # noqa: E402
import auditar_distratores as ad  # noqa: E402
import diversidade as dv  # noqa: E402
import schema_utils as su  # noqa: E402
from perfil_base import SAIDA_DIR, hash_questao, origem_de  # noqa: E402

V5 = ROOT / "data" / "train_curado_v5.jsonl"
V6 = ROOT / "data" / "train_curado_v6.jsonl"
INJECAO = ROOT / "data" / "injecao_v6.jsonl"
RESOLUCOES = SAIDA_DIR / "resolucoes_novas.jsonl"
PROVENIENCIA = SAIDA_DIR / "proveniencia_v6.jsonl"
EDICOES = ROOT / "ciclo_v6" / "EDICOES_HUMANAS.json"
QUARENTENA = ROOT / "ciclo_v6" / "QUARENTENA_HUMANA.json"
CONCEITUAIS = {("2º", "H11"), ("2º", "H12"), ("5º", "H12"), ("5º", "H13"), ("5º", "H14"), ("9º", "H14"), ("9º", "H17")}
VALS = [ROOT / "data" / n for n in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl")]
PROTEGIDOS = ("train_curado.jsonl", "train.jsonl", "train_curado_v3.jsonl", "train_curado_v5.jsonl")
SEED_LETRAS = 20261011


def _linhas(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _qs(ex):
    return json.loads(ex["messages"][2]["content"])["questoes"]


def _set_qs(ex, qs):
    ex["messages"][2]["content"] = json.dumps({"questoes": qs}, ensure_ascii=False)


def _norm(t):
    return dv.normalizar_texto(t)


def sha_res(t):
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()[:12]


def resolucoes_aprovadas(caminho=RESOLUCOES):
    """hash_questao -> registro da reescrita APROVADA mais recente."""
    out = {}
    if Path(caminho).exists():
        for r in _linhas(caminho):
            if r.get("decisao") == "aprovada" and r.get("resolucao_nova"):
                out[r["hash_questao"]] = r
    return out


def val_enunciados():
    s = set()
    for p in VALS:
        if p.exists():
            for ex in _linhas(p):
                try:
                    s |= {_norm(q.get("enunciado", "")) for q in _qs(ex)}
                except (KeyError, ValueError):
                    pass
    return s


def duplicatas_removiveis(v5, pares_path):
    """hashes de questões NÃO reais, 'duplicata_provavel', a remover (o outro lado fica)."""
    if not Path(pares_path).exists():
        return {}
    rem = {}
    for p in _linhas(pares_path):
        if p["classe"] != "duplicata_provavel":
            continue
        a, b = p["a"], p["b"]
        a_real, b_real = a["origem"].startswith("real"), b["origem"].startswith("real")
        if a_real and b_real:
            continue
        alvo = b if (a_real or not b_real) else a
        if alvo["origem"].startswith("real"):
            continue
        rem[alvo["hash"]] = {"motivo": "duplicata_provavel", "par": (a if alvo is b else b)["codigo_item"],
                             "jaccard": p["jaccard_trigramas"]}
    return rem


def montar(v5=V5, injecao=INJECAO, saida=V6, resolucoes=RESOLUCOES, prov=PROVENIENCIA, sem_gravar=False):
    saida = Path(saida)
    if saida.name in PROTEGIDOS or saida.name.startswith("val") or saida.resolve() == Path(v5).resolve():
        raise SystemExit(f"ABORTADO: não escrevo em {saida.name}.")
    base = _linhas(v5)
    apro = resolucoes_aprovadas(resolucoes)
    pares = SAIDA_DIR / "near_dups_v5.jsonl"
    dup = duplicatas_removiveis(base, pares)
    revisao_humana = {}
    dist = SAIDA_DIR / "distratores_v5_itens.jsonl"
    if dist.exists():
        for x in _linhas(dist):
            if x["revisao"] and not x["origem"].startswith("real"):
                revisao_humana[x["hash"]] = ",".join(m for m in ("alternativas_repetidas", "gabarito_com_valor_repetido",
                                                                  "fracoes_equivalentes_ao_gabarito") if x[m])
    quar = json.loads(QUARENTENA.read_text(encoding="utf-8"))["itens"] if QUARENTENA.exists() else {}
    edicoes = json.loads(EDICOES.read_text(encoding="utf-8"))["edicoes"] if EDICOES.exists() else {}
    registros, saida_ex = [], []

    # ---- v5 -> mantidas / aprimoradas / removidas
    for ex in base:
        qs = _qs(ex)
        cod = ex["meta"]["codigo_item"]
        hs = [hash_questao(q) for q in qs]
        qh = quar.get(cod)
        if qh and len(qs) == 1 and hs[0] in (qh.get("hash_v5"), qh.get("hash")):
            registros.append({"codigo_item": cod, "hash": hs[0], "operacao": "quarentena", "motivo": "amostra_humana:" + qh["motivo"]})
            continue
        ed = edicoes.get(cod)
        if ed and len(qs) == 1 and hs[0] == ed["hash_v5"] and not origem_de(ex["meta"]).startswith("real"):
            novo = json.loads(json.dumps(ex))
            q = dict(qs[0])
            antes_q = {"alternativas": dict(q["alternativas"]), "resolucao_passo_a_passo": q["resolucao_passo_a_passo"]}
            gab = q["alternativas"][q["resposta_correta"]]
            q["alternativas"] = dict(ed["alternativas"])
            q["resolucao_passo_a_passo"] = ed["resolucao_passo_a_passo"]
            assert q["alternativas"][q["resposta_correta"]] == gab, "a edição humana não pode mudar o gabarito"
            _set_qs(novo, [q])
            novo["meta"]["origem_v6"] = "v5"
            novo["meta"]["decisao_humana"] = {"acao": "manter_editar", "data": ed["data"], "autor": ed["autor"],
                                              "motivo": ed["motivo"], "campos_editados": ed["campos_editados"]}
            saida_ex.append(novo)
            registros.append({"codigo_item": cod, "hash": hs[0], "hash_v6": hash_questao(q), "operacao": "editada_por_decisao_humana",
                              "motivo": ed["motivo"], "antes": antes_q})
            continue
        if len(qs) == 1 and hs[0] in revisao_humana:
            registros.append({"codigo_item": cod, "hash": hs[0], "operacao": "quarentena",
                              "motivo": "revisao_humana:" + revisao_humana[hs[0]]})
            continue
        if len(qs) == 1 and hs[0] in dup:
            registros.append({"codigo_item": cod, "hash": hs[0], "operacao": "removida_duplicata", **dup[hs[0]]})
            continue
        novo = json.loads(json.dumps(ex))
        nqs, ops = [], []
        for q, h in zip(qs, hs):
            q = dict(q)
            if h in apro:
                r = apro[h]
                assert q.get("resolucao_passo_a_passo", "") == r["resolucao_original"], "v5 mudou sob a reescrita"
                q["resolucao_passo_a_passo"] = r["resolucao_nova"]
                novo["meta"]["resolucao_reescrita"] = {"modelo": r["modelos"]["professor_resolucao"],
                                                       "avaliador": r["modelos"]["avaliador_resolucao"],
                                                       "versao_prompt": r["versoes_prompt"]["professor_resolucao"],
                                                       "sha_original": sha_res(r["resolucao_original"])}
                ops.append(("resolucao_aprimorada", r))
            else:
                ops.append(("mantida", None))
            nqs.append(q)
        _set_qs(novo, nqs)
        novo["meta"]["origem_v6"] = "v5"
        saida_ex.append(novo)
        for (op, r), h in zip(ops, hs):
            reg = {"codigo_item": cod, "hash": h, "operacao": op}
            if r:
                reg.update(resolucao_original=r["resolucao_original"], resolucao_nova=r["resolucao_nova"],
                           modelos=r["modelos"], versoes_prompt=r["versoes_prompt"])
            registros.append(reg)

    # ---- injetadas
    val = val_enunciados()
    vistos_enun = {_norm(q.get("enunciado", "")) for ex in saida_ex for q in _qs(ex)}
    vistos_assin = {}
    for ex in saida_ex:
        for q in _qs(ex):
            s = aq.assinatura_dados(q)
            if s is not None:
                vistos_assin.setdefault(s, ex["meta"]["codigo_item"])
    por_hab = defaultdict(list)
    for ex in saida_ex:
        for q in _qs(ex):
            por_hab[(ex["meta"]["ano"], ex["meta"]["habilidade"])].append(q)
    novos = []
    for ex in _linhas(injecao):
        m = ex["meta"]
        cod = m["codigo_item"]
        q = _qs(ex)[0]
        h = hash_questao(q)
        chave = (m["ano"], m["habilidade"])
        motivo = None
        if cod in quar and h in (quar[cod].get("hash"), quar[cod].get("hash_v5")):
            registros.append({"codigo_item": cod, "hash": h, "operacao": "quarentena", "motivo": "amostra_humana:" + quar[cod]["motivo"]})
            continue
        if chave in CONCEITUAIS and (m.get("vereditos") or {}).get("revisor2") is not True:
            registros.append({"codigo_item": cod, "hash": h, "operacao": "quarentena",
                              "motivo": "conceitual_sem_segundo_revisor"})
            continue
        ind = ad.avaliar_questao(q)
        ok, _s, _mot = su.check_consistency_detalhado(q)
        enun = _norm(q.get("enunciado", ""))
        s = aq.assinatura_dados(q)
        if ind["revisao"]:
            registros.append({"codigo_item": cod, "hash": h, "operacao": "quarentena", "motivo": "revisao_humana:alternativas"})
            continue
        if ok is False or su.resposta_fora_das_alternativas(q):
            motivo = "verificador_reprova"
        elif enun in vistos_enun:
            motivo = "enunciado_duplicado"
        elif enun in val:
            motivo = "contaminacao_val"
        elif s is not None and s in vistos_assin:
            motivo = f"duplicata_semantica_de:{vistos_assin[s]}"
        elif any(dv.e_near_duplicata(q, o) for o in por_hab[chave]):
            motivo = "near_duplicata"
        if motivo:
            registros.append({"codigo_item": cod, "hash": h, "operacao": "descartada_na_montagem", "motivo": motivo})
            continue
        ex = json.loads(json.dumps(ex))
        ex["meta"]["origem_v6"] = "injecao_v6"
        novos.append(ex)
        vistos_enun.add(enun)
        if s is not None:
            vistos_assin[s] = cod
        por_hab[chave].append(q)

    # ---- letras: balanceia só entre as injetadas se a base passar de 19-21%
    def letras(exs):
        c = Counter(q["resposta_correta"] for e in exs for q in _qs(e))
        n = sum(c.values())
        return {L: round(100 * c.get(L, 0) / n, 2) for L in "ABCDE"}
    antes = letras(saida_ex + novos)
    if max(abs(v - 20) for v in antes.values()) > 1.0:
        from gerar_lote import permutar_lote
        flat = [_qs(e)[0] for e in novos]
        # compensa o desvio da base já fixa: ordem de preferência não usada; o guloso equilibra o lote novo
        for e, q in zip(novos, permutar_lote(flat, SEED_LETRAS)):
            _set_qs(e, [q])
    depois = letras(saida_ex + novos)
    for e in novos:
        registros.append({"codigo_item": e["meta"]["codigo_item"], "hash": hash_questao(_qs(e)[0]), "operacao": "injetada",
                          "modelos": e["meta"].get("modelos"), "versao_prompts": e["meta"].get("versao_prompts"),
                          "versao_gerador_v6": e["meta"].get("versao_gerador_v6"), "vereditos": e["meta"].get("vereditos"),
                          "hash_questao_injecao": e["meta"].get("hash_questao")})
    final = saida_ex + novos
    codigos = [e["meta"]["codigo_item"] for e in final]
    resumo = {"v5_exemplos": len(base), "v6_exemplos": len(final), "v6_questoes": sum(len(_qs(e)) for e in final),
              "operacoes": dict(Counter(r["operacao"] for r in registros)),
              "quarentena_motivos": dict(Counter(r["motivo"].split(":")[0] for r in registros if r["operacao"] == "quarentena")),
              "descartadas_motivos": dict(Counter(r["motivo"].split(":")[0] for r in registros if r["operacao"] == "descartada_na_montagem")),
              "letras_antes_do_balanceamento": antes, "letras_depois": depois,
              "codigos_repetidos": len(codigos) - len(set(codigos))}
    if not sem_gravar:
        saida.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in final), encoding="utf-8")
        Path(prov).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
    return resumo, final, registros


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default=str(V6))
    ap.add_argument("--injecao", default=str(INJECAO), help="arquivo de injetadas (padrão: rodadas 1 e 2; v6b = prompt reajustado)")
    ap.add_argument("--simular", action="store_true", help="calcula e imprime, sem gravar")
    args = ap.parse_args()
    resumo, _f, _r = montar(saida=args.saida, injecao=args.injecao, sem_gravar=args.simular)
    print(json.dumps(resumo, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
