"""Tarefa F, item 3: conjunto rotulado (itens avaliados pelo professor) em um único JSONL.

Junta cada folha preenchida com a respectiva chave_oculta.json e, quando a folha preenchida
traz só os rótulos (amostras v6), com a folha de entrega que tem o texto. Nada é descartado em
silêncio: o que não foi possível juntar vai para `nao_juntados` no relatório.

Regras de procedência (decididas ao ler os arquivos):
  * os ids opacos (Qnnnn) COLIDEM entre amostras; a chave de um item é (amostra, id);
  * `folha_preenchida_LLM.csv` (amostra_base_v7b) foi preenchida por LLM, NÃO pelo professor:
    é ignorada (a apuração do professor, apuracao.json, bate com folha_preenchida.csv);
  * `amostra_dificuldade*` usam outra rubrica (dificuldade_professor), não a de núcleo: fora;
  * a folha do Gemma s44 não está mais em ~/Downloads (foi para a lixeira); usa-se o primeiro
    candidato existente cujos ids/enunciados coincidam com a chave e cuja contagem de núcleo
    reproduza a apuracao.json.

Uso:  venv/bin/python src/f_rotulados.py
"""
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f_comum import (LETRAS, RAIZ, ROTULOS, SAIDA, assinatura, ler_csv, questao_de_linha,  # noqa: E402
                     sn, status_verificador, unidade_da_linha, valida_nucleo)

HOME = Path.home()

# amostra -> especificação. `bracos`: origem na chave -> (braço, família).
FONTES = [
    dict(amostra="amostra_3bracos", dir="outputs/amostra_3bracos",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json",
         bracos={"ft17b": ("1.7B ft v7b_s44", "qwen_1.7b_ft"),
                 "ft4b": ("4B ft", "qwen_4b_ft"),
                 "zs4b": ("4B zs", "qwen_4b_zs")}),
    dict(amostra="amostra_qwen4b", dir="outputs/amostra_qwen4b",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json",
         bracos={"finetune": ("1.7B ft v7b_s44", "qwen_1.7b_ft"),
                 "qwen4b": ("4B zs", "qwen_4b_zs")}),
    dict(amostra="amostra_humana", dir="outputs/amostra_humana",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json", texto="folha_professor_2.csv",
         bracos={"baseline": ("1.7B baseline_v1", "qwen_1.7b_baseline"),
                 "candidato": ("1.7B ft v7b_s44", "qwen_1.7b_ft")}),
    dict(amostra="amostra_humana_v7b_vs_baseline", dir="outputs/amostra_humana_v7b_vs_baseline",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json", texto="folha_professor.csv",
         bracos={"baseline": ("1.7B baseline_v1", "qwen_1.7b_baseline"),
                 "candidato": ("1.7B ft v7b", "qwen_1.7b_ft")}),
    dict(amostra="gemma3_s44", dir="outputs/gemma3/amostra_humana_s44",
         rotulos=[HOME / "Downloads" / "folha_preenchida.csv",
                  HOME / ".local/share/Trash/files/folha_preenchida.2.csv"],
         chave="chave_oculta.json", texto="folha_professor.csv",
         bracos={"baseline": ("gemma3 zs", "gemma_zs"),
                 "candidato": ("gemma3 ft s44", "gemma_ft")}),
    dict(amostra="provas_4b_prova_01", dir="outputs/provas_4b_20261019_0654/prova_01_numeros",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json",
         bracos={"qwen4b_ft": ("4B ft", "qwen_4b_ft")}),
    dict(amostra="amostra_base_v7b", dir="outputs/amostra_base_v7b",
         rotulos="folha_preenchida.csv", chave="chave_oculta.json",
         bracos={"banco": ("base-real banco SAEB", "base_corpus"),
                 "injecao": ("base injecao", "base_corpus"),
                 "L2": ("base L2", "base_corpus"),
                 "destilado": ("base destilado", "base_corpus")}),
    dict(amostra="amostra_humana_v6", dir="outputs/amostra_humana_v6",
         rotulos="interno/folha_preenchida_reduzida.csv", chave="interno/chave_oculta.json",
         texto="entregar/folha_professor.csv",
         bracos={"injetada": ("base v6 injetada", "base_corpus"),
                 "reescrita": ("base v6 reescrita", "base_corpus")}),
    dict(amostra="amostra_humana_v6_aderencia", dir="outputs/amostra_humana_v6_aderencia",
         rotulos="interno/folha_preenchida_reduzida.csv", chave="interno/chave_oculta.json",
         texto="entregar/folha_professor.csv",
         bracos={"estrutural": ("base v6 estrutural", "base_corpus"),
                 "controle": ("base v6 controle", "base_corpus")}),
    dict(amostra="amostra_humana_v6_controle", dir="outputs/amostra_humana_v6_controle",
         rotulos="interno/folha_preenchida_reduzida.csv", chave="interno/chave_oculta.json",
         texto="entregar/folha_professor.csv",
         bracos={"controle_v5": ("base v5 controle", "base_corpus")}),
]

# Amostras conhecidas que NÃO entram, com o motivo (vão para o relatório).
EXCLUIDAS = [
    ("outputs/amostra_base_v7b/folha_preenchida_LLM.csv", "preenchida por LLM, não pelo professor"),
    ("outputs/amostra_dificuldade/", "rubrica de dificuldade (dificuldade_professor), não a de núcleo"),
    ("outputs/amostra_dificuldade_ancora/", "rubrica de dificuldade, não a de núcleo"),
    ("outputs/amostra_humana_v6_promocao/", "só folha_professor.csv e apuracao.json agregada; folha preenchida não existe no repo"),
    ("outputs/amostra_humana_v6_vs_baseline/", "só folha_professor_2.csv e apuracao.json agregada; folha preenchida não existe no repo"),
    ("outputs/hf_4b/avaliacao/amostra_humana_3bracos_folha_preenchida.csv", "cópia byte a byte de amostra_3bracos/folha_preenchida.csv"),
    ("outputs/provas_4b_20261019_0654/prova_02..20", "sem rótulos do professor (pool pendente, item 1)"),
    ("outputs/folha_professor_ft4b_40", "sem rótulos do professor (pool pendente, item 1)"),
]


def _abs(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


def _texto_por_id(caminho):
    return {l["id"]: l for l in ler_csv(caminho)} if caminho and Path(caminho).exists() else {}


def _reproduz_apuracao(linhas_validas, apuracao_path):
    """(ok, detalhe): o núcleo recalculado bate com a apuração do professor?"""
    if not apuracao_path.exists():
        return None, "sem apuracao.json"
    ap = json.loads(apuracao_path.read_text(encoding="utf-8"))
    n = len(linhas_validas)
    s_nuc = sum(1 for l in linhas_validas if l["valida_nucleo"])
    s_ger = sum(1 for l in linhas_validas if l["valida_geral"])
    esp_ger = (ap.get("geral", {}).get("validade") or {}).get("s")
    esp_nuc = (ap.get("geral", {}).get("validade_nucleo") or {}).get("s")
    ok = (ap.get("n_avaliadas") == n and esp_ger == s_ger
          and (esp_nuc is None or esp_nuc == s_nuc))
    return ok, f"recalculado n={n} geral={s_ger} nucleo={s_nuc}; apuracao n={ap.get('n_avaliadas')} geral={esp_ger} nucleo={esp_nuc}"


def _escolhe_rotulos(fonte, base, chave, texto):
    """Primeiro candidato de folha preenchida que existe e bate com a chave (ids e enunciados)."""
    cands = fonte["rotulos"] if isinstance(fonte["rotulos"], list) else [fonte["rotulos"]]
    avisos = []
    for c in cands:
        p = _abs(c, base)
        if not p.exists():
            avisos.append(f"{p}: não existe")
            continue
        linhas = ler_csv(p)
        ids = {l.get("id", "").strip() for l in linhas}
        if not ids <= set(chave["itens"]) or len(ids) != len(chave["itens"]):
            avisos.append(f"{p}: ids não coincidem com a chave ({len(ids & set(chave['itens']))}/{len(chave['itens'])})")
            continue
        if texto and "enunciado" in linhas[0]:
            diverg = sum(1 for l in linhas
                         if (texto.get(l["id"], {}).get("enunciado") or "").strip() != (l.get("enunciado") or "").strip())
            if diverg:
                avisos.append(f"{p}: {diverg} enunciados divergem da folha de entrega")
                continue
        return p, linhas, avisos
    return None, [], avisos


def carregar_rotulados(raiz=RAIZ):
    """Devolve (linhas, relatorio). Pura em relação à rede; só lê arquivos."""
    raiz = Path(raiz)
    itens, rel_fontes, nao_juntados = [], [], []
    for fonte in FONTES:
        base = raiz / fonte["dir"]
        chave_p = base / fonte["chave"]
        if not chave_p.exists():
            rel_fontes.append({"amostra": fonte["amostra"], "status": "chave ausente", "n": 0})
            continue
        chave = json.loads(chave_p.read_text(encoding="utf-8"))
        texto_p = base / fonte["texto"] if fonte.get("texto") else None
        texto = _texto_por_id(texto_p)
        rot_p, linhas, avisos = _escolhe_rotulos(fonte, base, chave, texto)
        if rot_p is None:
            rel_fontes.append({"amostra": fonte["amostra"], "status": "folha preenchida não encontrada",
                               "avisos": avisos, "n": 0})
            continue
        validas = []
        for l in linhas:
            qid = (l.get("id") or "").strip()
            meta = chave["itens"][qid]
            origem = meta.get("origem") or meta.get("modelo")
            braco, familia = fonte["bracos"].get(origem, (f"desconhecido:{origem}", "desconhecido"))
            if braco.startswith("desconhecido"):
                nao_juntados.append({"amostra": fonte["amostra"], "id": qid, "motivo": f"origem {origem!r} sem braço mapeado"})
            fonte_texto = l if l.get("enunciado") else texto.get(qid, {})
            if not fonte_texto.get("enunciado"):
                nao_juntados.append({"amostra": fonte["amostra"], "id": qid, "motivo": "sem texto da questão"})
                continue
            rotulos = {k: sn(l.get(k)) for k in ROTULOS}
            if any(v is None for v in rotulos.values()):
                nao_juntados.append({"amostra": fonte["amostra"], "id": qid,
                                     "motivo": "rótulo vazio/inválido: " + ",".join(k for k, v in rotulos.items() if v is None)})
                continue
            q = questao_de_linha(fonte_texto)
            ano = fonte_texto.get("ano") or meta.get("ano")
            hab = fonte_texto.get("habilidade") or meta.get("habilidade")
            itens.append(OrderedDict([
                ("uid", f"{fonte['amostra']}:{qid}"), ("amostra", fonte["amostra"]), ("id_amostra", qid),
                ("braco", braco), ("familia", familia), ("origem_chave", origem),
                ("enunciado", q["enunciado"]),
                ("alternativas", q["alternativas"]),
                ("resolucao", q["resolucao_passo_a_passo"]),
                ("gabarito_modelo", q["resposta_correta"]),
                ("ano", ano), ("habilidade", hab),
                ("unidade", meta.get("unidade") or unidade_da_linha(ano, hab)),
                ("dificuldade_pedida", fonte_texto.get("dificuldade_pedida") or meta.get("dificuldade")),
                *[(k, rotulos[k]) for k in ROTULOS],
                ("valida_nucleo", valida_nucleo(rotulos)),
                ("comentario", l.get("comentario", "")),
                ("status_verificador", status_verificador(q)),
                ("status_chave", meta.get("status")),
                ("sig", assinatura(q)),
                ("fonte_rotulos", str(rot_p)),
            ]))
            validas.append(itens[-1])
        ok, det = _reproduz_apuracao(validas, base / "apuracao.json" if (base / "apuracao.json").exists()
                                     else base / "interno" / "apuracao.json")
        rel_fontes.append({"amostra": fonte["amostra"], "status": "ok", "n": len(validas),
                           "n_linhas_folha": len(linhas), "fonte_rotulos": str(rot_p),
                           "reproduz_apuracao_do_professor": ok, "detalhe_apuracao": det, "avisos": avisos})
    marcar_duplicados(itens)
    return itens, {"fontes": rel_fontes, "nao_juntados": nao_juntados,
                   "excluidas": [{"caminho": c, "motivo": m} for c, m in EXCLUIDAS]}


def marcar_duplicados(itens):
    """Acrescenta `dup_de` (uid da 1ª ocorrência da mesma questão) e `conflito_rotulo` in-place."""
    primeiro = {}
    for it in itens:
        p = primeiro.setdefault(it["sig"], it)
        it["dup_de"] = None if p is it else p["uid"]
        it["conflito_rotulo"] = (p is not it) and (p["valida_nucleo"] != it["valida_nucleo"])
    return itens


def resumo(itens):
    """Contagens por braço/família e taxas-base de cada dimensão (linhas únicas e todas)."""
    def taxas(grupo):
        n = len(grupo)
        d = {"n": n}
        for k in (*ROTULOS, "valida_nucleo"):
            s = sum(1 for g in grupo if g[k])
            d[k] = {"s": s, "pct": round(100 * s / n, 1) if n else None}
        return d
    unicos = [i for i in itens if i["dup_de"] is None]
    por_braco = OrderedDict((b, taxas([i for i in itens if i["braco"] == b]))
                            for b in sorted({i["braco"] for i in itens}))
    por_fam = OrderedDict((f, taxas([i for i in itens if i["familia"] == f]))
                          for f in sorted({i["familia"] for i in itens}))
    return {"n_linhas": len(itens), "n_unicos": len(unicos),
            "n_duplicados": len(itens) - len(unicos),
            "n_duplicados_com_rotulo_conflitante": sum(1 for i in itens if i["conflito_rotulo"]),
            "taxas_todas_as_linhas": taxas(itens), "taxas_unicos": taxas(unicos),
            "por_braco": por_braco, "por_familia": por_fam,
            "status_verificador": dict(Counter(i["status_verificador"] for i in itens))}


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    itens, rel = carregar_rotulados()
    with open(SAIDA / "rotulados.jsonl", "w", encoding="utf-8") as f:
        for it in itens:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rel["resumo"] = resumo(itens)
    (SAIDA / "rotulados_relatorio.json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    r = rel["resumo"]
    print(f"rotulados.jsonl: {r['n_linhas']} linhas ({r['n_unicos']} únicas, {r['n_duplicados']} duplicatas)")
    for f in rel["fontes"]:
        print(f"  {f['amostra']:<34} n={f['n']:<3} apuracao_reproduzida={f.get('reproduz_apuracao_do_professor')}")
    for b, t in r["por_braco"].items():
        print(f"  {b:<24} n={t['n']:<3} núcleo={t['valida_nucleo']['pct']}%")
    print("não juntados:", len(rel["nao_juntados"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
