"""Auditoria de gabarito de TODA a base de treino, em camadas, sem nenhuma chamada paga.

Pergunta que responde: para cada questão de data/train_curado_v3.jsonl (e, onde existe,
do DB/questoes.db), o gabarito está confirmado, refutado ou só aprovado por juiz?

Camadas, da mais forte para a mais fraca (a primeira que decide fixa a classe):
  1. REFUTADO   — o verificador aritmético estendido (schema_utils) acusa o gabarito, o
                  resultado da resolução não está em nenhuma alternativa, ou o gabarito do
                  treino difere do gabarito do banco oficial (DB/questoes.db).
  2. CONFIRMADO — (a) o verificador estendido confirma o gabarito; ou (b) as justificativas
                  oficiais do banco marcam "Correto" exatamente a alternativa do gabarito;
                  ou (c) item sintético (resposta calculada em Python).
  3. JUIZES     — sem confirmação determinística, mas os dois juízes cegos do pipeline
                  (validador + revisor, sabia-4-thinking, cada um RESOLVE o item sem ver o
                  gabarito) aprovaram: confiança "alta" em data/auditoria_base.jsonl, ou item
                  injetado (só entra com os dois vereditos true).
  3b. CEGO      — resolvido às cegas e confirmado em 2 passes (sabia-4-thinking); vale tanto
                  quanto JUIZES e é a evidência dos itens que os juízes deixaram "baixa/media".
  4. FRACO      — juízes "media"/"baixa", ou item nunca julgado com os prompts atuais.
  5. RESIDUAL   — não é REFUTADO nem CONFIRMADO e também não é JUIZES: são os únicos
                  candidatos a uma resolução cega paga. Os itens FRACO também entram.

Limite que o relatório declara: CONFIRMADO só certifica a aritmética (ou a concordância com
a justificativa oficial); JUIZES são do mesmo modelo e podem errar juntos. Nada aqui altera
nenhum gabarito: saída só em outputs/relatorios/.

Uso:
    venv/bin/python src/auditar_gabaritos_base.py [--base data/train_curado_v3.jsonl]
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import schema_utils as su  # noqa: E402
from unidades_tematicas import unidade_de  # noqa: E402

SAIDA_JSONL = ROOT / "outputs" / "relatorios" / "AUDITORIA_GABARITOS_v3.jsonl"
SAIDA_MD = ROOT / "outputs" / "relatorios" / "AUDITORIA_GABARITOS_v3.md"


def origem(codigo):
    if codigo.startswith("DIST-"):
        return "destilado"
    if codigo.startswith("INJ-"):
        return "injetado"
    if codigo.startswith("SINT-"):
        return "sintetico"
    if "-L2-2026-09" in codigo:
        return "real_L2"
    return "real_banco"


def carregar_banco(db=ROOT / "DB" / "questoes.db"):
    """codigo_item -> (gabarito, letra marcada 'Correto' nas justificativas ou None, {letra: texto}).

    Os textos permitem comparar o gabarito por CONTEÚDO: depois do balanceamento de
    letras (permutar_lote) a letra do treino deixa de ser a do banco, mas o texto da
    alternativa correta continua sendo o mesmo."""
    out = {}
    if not Path(db).exists():
        return out
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    for x in con.execute("select * from itens where gabarito in ('A','B','C','D')"):
        j = {L: (x[f"justificativa_alternativa_{L.lower()}"] or "").strip() for L in "ABCD"}
        marc = [L for L, v in j.items() if v.lower().startswith(("correto", "correta"))]
        textos = {L: (x[f"alternativa_{L.lower()}"] or "") for L in "ABCD"}
        out[x["codigo_item"]] = (x["gabarito"], marc[0] if len(marc) == 1 else None, textos)
    return out


def carregar_juizes(caminho=ROOT / "data" / "auditoria_base.jsonl"):
    """codigo_item -> confiança do registro mais recente."""
    ult = {}
    if Path(caminho).exists():
        for l in open(caminho, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("codigo_item"):
                    ult[r["codigo_item"]] = r
    return ult


_DATA_DE_PLANILHA = re.compile(r"^\d{4}-\d{2}-\d{2}( 00:00:00)?$")


def _canon(t):
    """forma_canonica + as duas diferenças legítimas entre o banco e o treino: o extrator troca
    "×" por "x" e os travessões ("−", "–", "—") por "-"."""
    from padronizar_enunciado import forma_canonica
    return forma_canonica(str(t).replace("×", "x").replace("−", "-").replace("–", "-").replace("—", "-"))


def _mesmo_conteudo(q, db, letra_banco):
    """A alternativa correta do treino tem o mesmo texto que a alternativa `letra_banco` do banco.
    Sem textos do banco (tupla de 2) ou se a alternativa foi reescrita (padronização só muda
    caixa/acento, que forma_canonica ignora), cai na comparação por letra."""
    if len(db) < 3 or not db[2].get(letra_banco):
        return q["resposta_correta"] == letra_banco
    if _DATA_DE_PLANILHA.match(str(db[2][letra_banco]).strip()):
        # o banco guardou uma fração como data (3/4 -> 2025-04-03): não há texto para comparar
        return True
    return _canon(q["alternativas"].get(q["resposta_correta"], "")) == _canon(db[2][letra_banco])


def _diverge_do_banco(q, db):
    return not _mesmo_conteudo(q, db, db[0])


def carregar_cegos():
    """codigo_item confirmados por RESOLUÇÃO CEGA paga (sabia-4-thinking, 2 permutações das alternativas):
    reverificar_gabaritos (PROPOSTAS_..., classe confirmado_cego) e consolidar_base (resolver:
    manter / manter_resolucao_antiga / trocar_gabarito, que só trocam com confirmação aritmética)."""
    out = set()
    prop = ROOT / "outputs" / "relatorios" / "PROPOSTAS_CORRECAO_GABARITO.jsonl"
    if prop.exists():
        out |= {json.loads(l)["codigo_item"] for l in open(prop, encoding="utf-8")
                if l.strip() and json.loads(l)["classe"] == "confirmado_cego"}
    res = ROOT / "outputs" / "relatorios" / "consolidacao_resolver.json"
    if res.exists():
        out |= {c for c, d in json.loads(res.read_text(encoding="utf-8")).items()
                if d["decisao"] in ("manter", "manter_resolucao_antiga", "trocar_gabarito")}
    return out


def classificar(q, codigo, banco, juizes, cegos=frozenset()):
    """Devolve (classe, evidencias:list[str])."""
    ev = []
    ok, _sug, motivo = su.check_consistency_detalhado(q)
    fora = su.resposta_fora_das_alternativas(q)
    db = banco.get(codigo)
    org = origem(codigo)
    gab = q["resposta_correta"]
    if ok is False:
        ev.append(f"verificador:{motivo}")
    if fora:
        ev.append("resultado_fora_das_alternativas")
    if db and _diverge_do_banco(q, db):
        ev.append(f"gabarito_diverge_do_banco({db[0]}!={gab})")
    if ev:
        return "REFUTADO", ev
    if ok is True:
        ev.append(f"verificador:{motivo}")
    if db and db[1] and _mesmo_conteudo(q, db, db[1]):
        ev.append("justificativa_oficial_confirma")
    if org == "sintetico":
        ev.append("sintetico_calculado")
    if ev:
        return "CONFIRMADO", ev
    j = juizes.get(codigo)
    if codigo in cegos:
        return "CEGO", ["resolucao_cega_2_passes_confirma"]
    if org == "injetado":
        return "JUIZES", ["aprovado_nos_dois_juizes_na_injecao"]
    if j and j.get("confianca") == "alta":
        return "JUIZES", ["confianca_alta(validador+revisor)"]
    if j:
        return "FRACO", [f"confianca_{j.get('confianca')}"]
    return "FRACO", ["nunca_julgado_com_os_prompts_atuais"]


def auditar(base, banco=None, juizes=None, cegos=None):
    cegos = carregar_cegos() if cegos is None else cegos
    banco = carregar_banco() if banco is None else banco
    juizes = carregar_juizes() if juizes is None else juizes
    itens = []
    for i, l in enumerate(open(base, encoding="utf-8")):
        if not l.strip():
            continue
        r = json.loads(l)
        m = r["meta"]
        for j, q in enumerate(json.loads(r["messages"][2]["content"])["questoes"]):
            classe, ev = classificar(q, m["codigo_item"], banco, juizes, cegos)
            itens.append({"linha": i, "questao": j, "codigo_item": m["codigo_item"], "ano": m["ano"],
                          "habilidade": m["habilidade"], "unidade": unidade_de(m["ano"], m["habilidade"]),
                          "origem": origem(m["codigo_item"]), "classe": classe, "evidencias": ev,
                          "gabarito": q["resposta_correta"],
                          "avisos": su.avisos_consistencia(q)})
    return itens


def resumo_md(itens, base):
    n = len(itens)
    ordem = ["CONFIRMADO", "JUIZES", "CEGO", "FRACO", "REFUTADO"]
    por = Counter(x["classe"] for x in itens)
    linhas = [f"# Auditoria de gabaritos — {Path(base).name}", "",
              f"{n} questões. Sem nenhuma chamada paga. Nada foi alterado.", "",
              "| Classe | Questões | % |", "|---|---|---|"]
    for c in ordem:
        linhas.append(f"| {c} | {por[c]} | {100 * por[c] / n:.1f} |")
    for titulo, chave in (("Por unidade temática", "unidade"), ("Por origem", "origem")):
        tab = defaultdict(Counter)
        for x in itens:
            tab[x[chave]][x["classe"]] += 1
        linhas += ["", f"## {titulo}", "", "| " + chave + " | n | " + " | ".join(ordem) + " |", "|---|---|" + "---|" * len(ordem)]
        for k in sorted(tab, key=str):
            t = tab[k]
            linhas.append(f"| {k} | {sum(t.values())} | " + " | ".join(str(t[c]) for c in ordem) + " |")
    ref = [x for x in itens if x["classe"] == "REFUTADO"]
    linhas += ["", f"## Refutados ({len(ref)})", ""]
    for x in ref[:60]:
        linhas.append(f"- {x['codigo_item']} ({x['ano']} {x['habilidade']}): {', '.join(x['evidencias'])}")
    fr = [x for x in itens if x["classe"] == "FRACO"]
    linhas += ["", f"## Candidatos à resolução cega paga (FRACO): {len(fr)}", "",
               "Motivos: " + ", ".join(f"{k}={v}" for k, v in Counter(e for x in fr for e in x["evidencias"]).most_common())]
    linhas += ["", "## Limites", "",
               "- CONFIRMADO certifica a aritmética (ou a concordância com a justificativa oficial), não a matemática inteira.",
               "- JUIZES são do mesmo modelo (sabia-4-thinking) e podem errar juntos.",
               "- Esta auditoria não vê erro de enunciado (pergunta errada com resolução coerente)."]
    return "\n".join(linhas) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v3.jsonl"))
    ap.add_argument("--rotulo", default="v3", help="sufixo dos arquivos de saída (AUDITORIA_GABARITOS_<rotulo>.*)")
    args = ap.parse_args()
    itens = auditar(args.base)
    global SAIDA_JSONL, SAIDA_MD
    SAIDA_JSONL = ROOT / "outputs" / "relatorios" / f"AUDITORIA_GABARITOS_{args.rotulo}.jsonl"
    SAIDA_MD = ROOT / "outputs" / "relatorios" / f"AUDITORIA_GABARITOS_{args.rotulo}.md"
    SAIDA_JSONL.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_JSONL.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in itens), encoding="utf-8")
    md = resumo_md(itens, args.base)
    SAIDA_MD.write_text(md, encoding="utf-8")
    print(md[:2500])


if __name__ == "__main__":
    main()
