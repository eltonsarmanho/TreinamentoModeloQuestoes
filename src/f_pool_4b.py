"""Tarefa F, itens 1 e 2: pool de itens do 4B fine-tunado ainda NÃO avaliados pelo professor e
uma folha cega única, em ordem estratificada, com lotes de 40.

Fontes (adapter lora_4b_v7b_s44, pipeline de produção, status `falha` já descartado na geração):
  outputs/folha_professor_ft4b_40/                          40 itens, folha não preenchida
  outputs/provas_4b_20261019_0654/prova_NN_*/folhaN_professor.csv   20 provas (a 01 já foi avaliada)

Item "avaliado" = qualquer coluna de rubrica preenchida na folha correspondente. Duplicata =
mesmo enunciado+alternativas (sem caixa/acento/espaço), dentro do pool OU contra qualquer item
já avaliado (outputs/melhoria_v8/F/rotulados.jsonl: 3 braços, qwen4b, prova_01, ...).

Ordem da folha: rodízio entre as 5 unidades temáticas, sorteio com seed fixa dentro de cada
unidade, e a ordem das unidades dentro de cada rodada também é sorteada. Logo TODO prefixo de
tamanho múltiplo de 5 (em particular 40, 80, 120...) tem a mesma quantidade de itens por
unidade, e o professor pode parar em qualquer múltiplo de 40.

Uso:  venv/bin/python src/f_pool_4b.py [--seed 20261019]
Não chama modelo nem API; só lê e escreve arquivos em outputs/melhoria_v8/F/.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversidade as dv  # noqa: E402
from f_comum import (LETRAS, RAIZ, ROTULOS, SAIDA, UNIDADES, assinatura, ler_csv,  # noqa: E402
                     questao_de_linha, status_verificador, unidade_da_linha)
from f_estatistica import tabela_tamanho_amostra  # noqa: E402

SEED_PADRAO = 20261019
TAMANHO_LOTE = 40
FOLHA_NOME = "folha_professor_4b_pendentes.csv"
# mesmas colunas e ordem das folhas existentes (amostra_humana.COLUNAS_FOLHA)
COLUNAS_FOLHA = (["id", "ano", "habilidade", "descricao_habilidade", "dificuldade_pedida", "enunciado"]
                 + list(LETRAS) + ["resolucao", "gabarito_modelo"] + list(ROTULOS) + ["comentario"])
PASTA_FT4B_40 = "outputs/folha_professor_ft4b_40"
PASTA_PROVAS = "outputs/provas_4b_20261019_0654"
INSTRUCOES_PADRAO = "outputs/amostra_3bracos/INSTRUCOES_PROFESSOR.md"


# --------------------------------------------------------------------------
# lógica pura (testada em tests/test_f_pool_4b.py)
# --------------------------------------------------------------------------
def linha_avaliada(linha):
    """True se o professor já preencheu algo da rubrica nessa linha."""
    return any((linha.get(k) or "").strip() for k in ROTULOS)


def separar_duplicatas(candidatos, sigs_avaliadas):
    """candidatos: lista ordenada de dicts com `sig`. Devolve (mantidos, descartados) onde cada
    descartado ganha `motivo_descarte` ('ja_avaliado' | 'duplicata_no_pool') e `duplica`."""
    mantidos, descartados, vistos = [], [], {}
    for c in candidatos:
        if c["sig"] in sigs_avaliadas:
            descartados.append({**c, "motivo_descarte": "ja_avaliado",
                                "duplica": sigs_avaliadas[c["sig"]]})
        elif c["sig"] in vistos:
            descartados.append({**c, "motivo_descarte": "duplicata_no_pool",
                                "duplica": vistos[c["sig"]]})
        else:
            vistos[c["sig"]] = c["origem_pasta"] + ":" + str(c["id_original"])
            mantidos.append(c)
    return mantidos, descartados


def ordem_estratificada(itens, seed, unidades=UNIDADES, chave=lambda it: it["unidade"]):
    """Rodízio entre unidades; sorteio com `seed` dentro de cada unidade e da ordem das unidades em
    cada rodada. Quando uma unidade acaba, o rodízio segue com as restantes (o prefixo deixa de
    ser exatamente balanceado só a partir daí). Itens sem unidade conhecida vão ao fim."""
    rng = random.Random(seed)
    filas = OrderedDict((u, []) for u in unidades)
    resto = []
    for it in sorted(itens, key=lambda x: (x["origem_pasta"], str(x["id_original"]))):
        u = chave(it)
        (filas[u] if u in filas else resto).append(it)
    for u in filas:
        rng.shuffle(filas[u])
    saida = []
    while any(filas.values()):
        ativas = [u for u in filas if filas[u]]
        rng.shuffle(ativas)
        for u in ativas:
            saida.append(filas[u].pop(0))
    rng.shuffle(resto)
    return saida + resto


def prefixo_balanceado(ordem, k, unidades=UNIDADES, chave=lambda it: it["unidade"]):
    """Contagem por unidade nos k primeiros itens."""
    c = Counter(chave(it) for it in ordem[:k])
    return {u: c.get(u, 0) for u in unidades}


def quase_duplicatas(itens):
    """Informativo: pares de itens que só diferem nos números (gabarito/contas podem diferir).
    Não descarta nada; devolve {n_grupos, n_itens_em_grupos}."""
    g = defaultdict(list)
    for it in itens:
        q = questao_de_linha(it["linha"])
        g[dv.mascarar_numeros(dv.texto_questao(q, com_alternativas=False))].append(it)
    grupos = [v for v in g.values() if len(v) > 1]
    return {"n_grupos": len(grupos), "n_itens_em_grupos": sum(len(v) for v in grupos)}


def atribuir_ids(n, seed):
    """Ids opacos 'Qnnnn', únicos, sem relação com a ordem."""
    rng = random.Random(seed + 7)
    return [f"Q{i}" for i in rng.sample(range(1000, 10000), n)]


def dividir_lotes(linhas, tamanho=TAMANHO_LOTE):
    return [linhas[i:i + tamanho] for i in range(0, len(linhas), tamanho)]


# --------------------------------------------------------------------------
# coleta
# --------------------------------------------------------------------------
def _pastas_fonte(raiz):
    pastas = [(Path(raiz) / PASTA_FT4B_40)]
    pastas += sorted(p for p in (Path(raiz) / PASTA_PROVAS).glob("prova_*") if p.is_dir())
    return pastas


def _folha_entrega(pasta):
    """folha_professor.csv ou folhaN_professor.csv (as provas 02..10 foram renomeadas)."""
    achados = sorted(pasta.glob("folha*_professor.csv"))
    return achados[0] if achados else None


def coletar_pool(raiz=RAIZ):
    """Todos os itens das folhas de entrega do 4B, com marca de avaliado/pendente.
    Devolve (itens, relatorio_fontes)."""
    itens, fontes = [], []
    for pasta in _pastas_fonte(raiz):
        folha = _folha_entrega(pasta)
        chave_p = pasta / "chave_oculta.json"
        if folha is None or not chave_p.exists():
            fontes.append({"pasta": str(pasta), "status": "sem folha ou chave"})
            continue
        chave = json.loads(chave_p.read_text(encoding="utf-8"))["itens"]
        # folha preenchida do professor (se existir): só vale se tiver rubrica preenchida
        preenchidas = [p for p in pasta.glob("folha*_preenchida.csv")]
        rot_preenchidos = set()
        for p in preenchidas:
            for l in ler_csv(p):
                if linha_avaliada(l):
                    rot_preenchidos.add(l["id"])
        linhas = ler_csv(folha)
        n_av = 0
        for l in linhas:
            meta = chave.get(l["id"], {})
            avaliada = linha_avaliada(l) or l["id"] in rot_preenchidos
            n_av += avaliada
            q = questao_de_linha(l)
            itens.append({
                "origem_pasta": str(pasta.relative_to(raiz)), "id_original": l["id"],
                "avaliada": avaliada, "linha": l, "sig": assinatura(q),
                "ano": l.get("ano"), "habilidade": l.get("habilidade"),
                "unidade": unidade_da_linha(l.get("ano"), l.get("habilidade")),
                "dificuldade": l.get("dificuldade_pedida"),
                "modelo": meta.get("modelo"), "status_chave": meta.get("status"),
                "seed_geracao": meta.get("seed"),
                "status_verificador": status_verificador(q),
                "sem_chave": l["id"] not in chave,
            })
        fontes.append({"pasta": str(pasta.relative_to(raiz)), "folha": folha.name,
                       "n_itens": len(linhas), "n_avaliados": n_av,
                       "n_com_rotulo_preenchido_em_arquivo": len(preenchidas)})
    return itens, fontes


def sigs_ja_avaliadas(raiz=RAIZ):
    """sig -> 'amostra:id' de tudo que tem rótulo do professor (rotulados.jsonl)."""
    p = Path(raiz) / "outputs" / "melhoria_v8" / "F" / "rotulados.jsonl"
    if not p.exists():
        raise SystemExit("rode antes: venv/bin/python src/f_rotulados.py (gera rotulados.jsonl)")
    out = {}
    for linha in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        out.setdefault(r["sig"], f"{r['amostra']}:{r['id_amostra']}")
    return out


# --------------------------------------------------------------------------
# saídas
# --------------------------------------------------------------------------
def montar_linha_folha(item, novo_id):
    base = {k: item["linha"].get(k, "") for k in COLUNAS_FOLHA}
    base["id"] = novo_id
    for k in (*ROTULOS, "comentario"):
        base[k] = ""
    if not base.get("descricao_habilidade"):
        try:
            h = dv.obter_habilidade(item["ano"], item["habilidade"])
            base["descricao_habilidade"] = (h or {}).get("descricoes", [""])[0] if h else ""
        except Exception:  # noqa: BLE001
            pass
    return base


def escrever_csv(caminho, linhas):
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_FOLHA)
        w.writeheader()
        w.writerows(linhas)


def instrucoes_do_professor(raiz, n_total, n_lotes):
    """Instruções padrão (rubrica idêntica) com título, contagem e devolução adaptados."""
    base = (Path(raiz) / INSTRUCOES_PADRAO).read_text(encoding="utf-8")
    base = base.replace("# Avaliação de questões de matemática (SAEB), 60 questões",
                        f"# Avaliação de questões de matemática (SAEB), {n_total} questões em {n_lotes} lotes de 40")
    ini = base.index("Obrigado por aceitar")
    fim = base.index("## O que você precisa fazer")
    intro = (f"Obrigado por aceitar avaliar. Há **{n_total} questões** de matemática do Ensino "
             f"Fundamental, divididas em **{n_lotes} lotes de 40** (`folha_lote_01.csv`, "
             "`folha_lote_02.csv`, ...). Tempo estimado: cerca de **2 minutos por questão** "
             "(uns 80 minutos por lote). **Você pode parar ao final de qualquer lote**: cada lote "
             "completo já vale por si. Comece sempre pelo lote 01 e siga a ordem.\n\n")
    base = base[:ini] + intro + base[fim:]
    base = base.replace("1. Abra **`folha_professor.csv`**", "1. Abra o próximo lote, por exemplo **`folha_lote_01.csv`**,")
    base = base.replace("4. Salve como **CSV UTF-8** com o nome **`folha_preenchida.csv`** e devolva ao \n   responsável técnico.",
                        "4. Salve como **CSV UTF-8** com o nome do lote mais `_preenchida`, por exemplo\n"
                        "   **`folha_lote_01_preenchida.csv`**, e devolva ao responsável técnico.\n"
                        "   Pode devolver cada lote assim que terminá-lo.")
    base = base.replace("Envie **somente** o arquivo `folha_preenchida.csv`.",
                        "Envie **somente** os arquivos `folha_lote_NN_preenchida.csv`.")
    base = base.replace("Pode reordenar ou filtrar as linhas, se ajudar.",
                        "Não reordene as linhas: a ordem dos lotes é parte do plano de amostragem.")
    return base


def readme(rel, tabela, raiz_rel="outputs/melhoria_v8/F"):
    p = rel["pool"]
    linhas_unid = "\n".join(f"| {u} | {rel['por_unidade_pendentes'].get(u, 0)} | "
                            f"{rel['por_unidade_total_coletado'].get(u, 0)} |" for u in UNIDADES)
    linhas_ano = "\n".join(f"| {a} | {n} |" for a, n in sorted(rel["por_ano_pendentes"].items()))
    linhas_tab = "\n".join(f"| {t['n']} | {t['ic_lo_pct']}% a {t['ic_hi_pct']}% | ±{t['meia_largura_pp']} pp |"
                           for t in tabela)
    n_lotes = rel["n_lotes"]
    return f"""# Folha cega do Qwen3-4B fine-tunado (adapter lora_4b_v7b_s44): itens pendentes

Gerado por `src/f_pool_4b.py` (seed {rel['seed']}). Objetivo: medir com precisão a validade de núcleo
(`correcao_matematica` ∧ `resposta_unica` ∧ `dados_suficientes`) do modelo atual. Hoje há só 30 itens do
4B fine-tunado avaliados (20 na amostra de 3 braços + 10 da prova 01): IC de ±{rel['meia_largura_30_pp']} pp.

## O que entregar ao professor (e o que NÃO entregar)

Entregar, **somente**:

- `folha_lote_01.csv` ... `folha_lote_{n_lotes:02d}.csv` (40 questões cada; o último pode ter menos)
- `INSTRUCOES_PROFESSOR.md` (mesma rubrica das folhas anteriores)
- (`{FOLHA_NOME}` é a folha inteira, equivalente à concatenação dos lotes; use uma OU a outra, não as duas)

**Não entregar**: este README, a pasta `chave/` (contém `chave_oculta.json`), `pool_4b_relatorio.json`.
As folhas não têm nenhum indício de origem: ids opacos `Qnnnn` sem relação com a prova de origem, com a
semente ou com o status do verificador; a chave só existe em `chave/`.

## O que o professor preenche

As 7 colunas em branco por linha, com `s` ou `n` (minúsculas): `correcao_matematica`, `resposta_unica`,
`dados_suficientes`, `aderencia_ano_habilidade`, `contexto_plausivel`, `dificuldade_adequada`,
`valida_geral`; `comentario` é opcional. A decisão é tomada só com as 3 primeiras (núcleo), mas as 7
precisam estar preenchidas para a linha contar. Tempo esperado: **~2 min por item**, ou seja ~80 min por
lote de 40. Devolver como `folha_lote_NN_preenchida.csv` (UTF-8).

## Ordem estratificada: pode parar em qualquer múltiplo de 40

Cada rodada de 5 linhas traz 1 item de cada unidade temática (Números, Álgebra, Geometria, Grandezas e
medidas, Probabilidade e estatística), em ordem sorteada; dentro da unidade o sorteio usa seed fixa. Logo
cada lote de 40 tem 8 itens de cada unidade e **qualquer prefixo de tamanho múltiplo de 5 é balanceado**.
Se o professor parar no meio do lote, as linhas completas ainda contam, mas só múltiplos de 5 mantêm o
balanço exato. A unidade menor ({rel['unidade_menor']}) tem {rel['n_unidade_menor']} itens pendentes; por isso
o balanço exato vale até a linha **{rel['balanceado_ate']}** (lotes 1 a {rel['balanceado_ate'] // 40} completos); depois disso o rodízio segue sem essa
unidade (contagens por prefixo em `pool_4b_relatorio.json`, campo `prefixos`).

## Como apurar (inclusive folha parcial)

Opção recomendada (aceita um ou vários lotes, ignora linhas em branco, confere o balanço do prefixo):

```bash
venv/bin/python src/f_apurar_lotes.py \\
    --lotes {raiz_rel}/folha_lote_01_preenchida.csv {raiz_rel}/folha_lote_02_preenchida.csv \\
    --chave {raiz_rel}/chave/chave_oculta.json --saida {raiz_rel}/apuracao_parcial.json
```

Equivalente com o `apurar` original (uma folha, linhas não preenchidas descartadas com
`--ignorar-incompletas`; imprime a lista das linhas em branco em stderr e sai com código 0):

```bash
venv/bin/python src/amostra_humana.py apurar --ignorar-incompletas \\
    --folha folha_professor_4b_pendentes_preenchida.csv \\
    --chave {raiz_rel}/chave/chave_oculta.json --saida apuracao.json
```

Sem `--ignorar-incompletas` o `apurar` original recusa (código 2) qualquer folha com linha em branco.
Ambos foram testados com uma folha fictícia em diretório temporário (`tests/test_f_pool_4b.py`).

## Tamanho de amostra: n avaliados -> largura do IC de Wilson 95% quando a validade observada é 65%

| n avaliados | IC 95% | meia-largura |
|---|---|---|
{linhas_tab}

Com os {rel['n_pendentes']} itens pendentes somados aos 30 do 4B já avaliados (n={rel['n_pendentes'] + 30})
chega-se a cerca de ±{rel['meia_largura_total_pp']} pp. Para comparar DUAS versões é preciso o IC de cada uma e
amostras pareadas ou muito maiores: este pool serve para medir UM modelo, o atual.

## Contagens do pool

- itens coletados nas 21 pastas: {p['coletados']}
- já avaliados pelo professor (prova_01, 10): {p['ja_avaliados_na_folha']}
- duplicados contra itens já avaliados (outros braços/amostras, mesmo enunciado+alternativas): {p['duplicados_de_avaliados']}
- duplicados dentro do pool: {p['duplicados_no_pool']}
- **pendentes únicos (esta folha): {rel['n_pendentes']}** em {n_lotes} lotes

| unidade temática | pendentes (folha) | coletados (antes de dedup) |
|---|---|---|
{linhas_unid}

| ano | pendentes |
|---|---|
{linhas_ano}

Dificuldade pedida: {rel['por_dificuldade_pendentes']}. Status do verificador determinístico atual:
{rel['por_status_verificador_pendentes']}.

## Arquivos

- `{FOLHA_NOME}`, `folha_lote_NN.csv`, `INSTRUCOES_PROFESSOR.md`: para o professor
- `chave/chave_oculta.json`: chave no formato do `amostra_humana.py apurar` (origem `ft4b`; guarda também a
  prova de origem, o id original, o seed de geração e o status da geração)
- `pool_4b_relatorio.json`: contagens, descartes e motivos
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED_PADRAO)
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args(argv)
    saida = Path(args.saida)
    (saida / "chave").mkdir(parents=True, exist_ok=True)

    todos, fontes = coletar_pool(RAIZ)
    avaliados_na_folha = [i for i in todos if i["avaliada"]]
    pendentes_brutos = [i for i in todos if not i["avaliada"]]
    mantidos, descartados = separar_duplicatas(pendentes_brutos, sigs_ja_avaliadas(RAIZ))

    ordem = ordem_estratificada(mantidos, args.seed)
    ids = atribuir_ids(len(ordem), args.seed)
    linhas = [montar_linha_folha(it, i) for it, i in zip(ordem, ids)]
    escrever_csv(saida / FOLHA_NOME, linhas)
    lotes = dividir_lotes(linhas)
    for k, lote in enumerate(lotes, 1):
        escrever_csv(saida / f"folha_lote_{k:02d}.csv", lote)

    chave = {
        "versao": 1, "aviso": "NÃO ENTREGAR AO PROFESSOR — quebra o cegamento.",
        "seed": args.seed, "n": len(ordem), "adapter": "lora_4b_v7b_s44",
        "itens": {i: {"origem": "ft4b", "relatorio": it["origem_pasta"], "item": it["id_original"],
                      "unidade": it["unidade"], "ano": it["ano"], "habilidade": it["habilidade"],
                      "status": it["status_chave"], "geometria": None, "seed_rodada": 0,
                      "dificuldade": it["dificuldade"], "seed_geracao": it["seed_geracao"],
                      "pos_na_folha": pos}
                  for pos, (it, i) in enumerate(zip(ordem, ids), 1)},
    }
    (saida / "chave" / "chave_oculta.json").write_text(
        json.dumps(chave, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "INSTRUCOES_PROFESSOR.md").write_text(
        instrucoes_do_professor(RAIZ, len(ordem), len(lotes)), encoding="utf-8")

    n_pend = len(ordem)
    ns = [20, 30, 40, 80, 120, 160, 200, 240, 300, 400]
    tabela = tabela_tamanho_amostra(ns)
    rel = {
        "seed": args.seed, "n_pendentes": n_pend, "n_lotes": len(lotes),
        "fontes": fontes,
        "pool": {"coletados": len(todos), "ja_avaliados_na_folha": len(avaliados_na_folha),
                 "duplicados_de_avaliados": sum(d["motivo_descarte"] == "ja_avaliado" for d in descartados),
                 "duplicados_no_pool": sum(d["motivo_descarte"] == "duplicata_no_pool" for d in descartados)},
        "descartados": [{"origem": d["origem_pasta"], "id": d["id_original"], "motivo": d["motivo_descarte"],
                         "duplica": d["duplica"]} for d in descartados],
        "por_unidade_pendentes": dict(Counter(i["unidade"] for i in mantidos)),
        "por_unidade_total_coletado": dict(Counter(i["unidade"] for i in todos)),
        "por_ano_pendentes": dict(Counter(i["ano"] for i in mantidos)),
        "por_dificuldade_pendentes": dict(Counter(i["dificuldade"] for i in mantidos)),
        "por_status_verificador_pendentes": dict(Counter(i["status_verificador"] for i in mantidos)),
        "por_status_geracao_pendentes": dict(Counter(i["status_chave"] for i in mantidos)),
        "prefixos": {str(k): prefixo_balanceado(ordem, k) for k in range(40, n_pend + 1, 40)},
        "sem_unidade": sum(1 for i in mantidos if i["unidade"] is None),
        "unidade_menor": min(UNIDADES, key=lambda u: sum(1 for i in mantidos if i["unidade"] == u)),
        "n_unidade_menor": min(sum(1 for i in mantidos if i["unidade"] == u) for u in UNIDADES),
        "balanceado_ate": 5 * min(sum(1 for i in mantidos if i["unidade"] == u) for u in UNIDADES),
        "quase_duplicatas_numeros_mascarados": quase_duplicatas(mantidos),
        "meia_largura_30_pp": tabela_tamanho_amostra([30])[0]["meia_largura_pp"],
        "tabela_tamanho_amostra_p65": tabela,
        "meia_largura_total_pp": tabela_tamanho_amostra([n_pend + 30])[0]["meia_largura_pp"],
    }
    (saida / "pool_4b_relatorio.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "README.md").write_text(readme(rel, tabela), encoding="utf-8")

    print(f"coletados {len(todos)}; já avaliados {len(avaliados_na_folha)}; "
          f"duplicados (avaliados/pool) {rel['pool']['duplicados_de_avaliados']}/{rel['pool']['duplicados_no_pool']}; "
          f"pendentes {n_pend} em {len(lotes)} lotes")
    print("por unidade:", rel["por_unidade_pendentes"])
    print("por ano:", rel["por_ano_pendentes"])
    print("prefixos (40,80,...):", {k: list(v.values()) for k, v in rel["prefixos"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
