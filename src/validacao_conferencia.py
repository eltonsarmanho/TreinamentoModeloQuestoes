"""Validação da conferência (cot / crítica) em folha NOVA do professor (ciclo v8, 2026-10-25).

A crítica v2 foi desenhada olhando os erros das 78 questões do v8rftb (85,2%, IC95 [73,4; 92,3]), então esse número
não vale como prova. Aqui medimos em questões que o método nunca viu, geradas com outro modelo (v7b) e outra seed, e
enviadas ao professor TODAS (mantidas e descartadas), às cegas. O critério abaixo foi fixado ANTES de existir dado
(ciclo_v8/SPEC.md) e o prompt está congelado por `tests/test_auto_conferencia.py::test_prompt_critica_congelado`.

  novidade  confere se alguma questão da folha nova repete o enunciado de folhas antigas (contaminação).
  apurar    folha preenchida + respostas da cot e da crítica -> tabela, IC95 de Wilson e veredito do critério.
            Exige respostas para TODAS as questões: apuração parcial enviesa (as que faltam não são aleatórias).
            Com --revisao/--mapa: apuração de SENSIBILIDADE com os rótulos da revisão cega (não substitui a oficial).
  revisao   folha cega com as 6 questões de rótulo suspeito + controles sorteados, ids novos, para o professor rever.
Sem API, sem GPU.

Uso:
  venv/bin/python src/validacao_conferencia.py novidade --folha <nova>/folha_professor.csv
  venv/bin/python src/validacao_conferencia.py apurar --folha <preenchida>.csv --chave <nova>/chave_oculta.json --rotulo val_v7b
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))
import auto_conferencia as ac  # noqa: E402

SAIDA = ROOT / "outputs" / "melhoria_v8" / "validacao_conferencia"
ANTIGAS = [ROOT / "outputs" / "melhoria_v8" / "folha_v8rftb" / "folha_preenchida.csv",
           ROOT / "outputs" / "melhoria_v8" / "F" / "folha_lote_01_preenchida.csv"]

# Fixado em 2026-10-25, antes de gerar a folha. Mudar depois de ver o resultado invalida a validação.
CRITERIO = {"nucleo_min_pct": 80.0, "ic_inferior_min_pct": 70.0, "validas_perdidas_max_pct": 15.0, "ganho_min_pp": 10.0}


def normalizar(texto):
    t = unicodedata.normalize("NFKD", texto or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def enunciados(folha):
    with open(folha, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: normalizar(r["enunciado"]) for r in csv.DictReader(f)}


def duplicatas(novas, antigas):
    """ids da folha nova cujo enunciado (sem acento, caixa ou pontuação) já aparece em folha antiga."""
    vistos = set(antigas.values())
    return sorted(i for i, e in novas.items() if e in vistos)


def linha(itens, mantem):
    """Quadro de uma regra: quantas mantém, núcleo entre as mantidas (Wilson) e quantas válidas perde."""
    m = [x for x in itens if mantem(x)]
    v = sum(bool(x["valida_nucleo"]) for x in itens)
    vm = sum(bool(x["valida_nucleo"]) for x in m)
    ic = ac.wilson(vm, len(m)) if m else (None, None)
    return {"mantidas": len(m), "mantidas_pct": round(100 * len(m) / len(itens), 1),
            "nucleo": f"{vm}/{len(m)}", "nucleo_pct": round(100 * vm / len(m), 1) if m else None,
            "ic95": list(ic), "validas_perdidas": f"{v - vm}/{v}",
            "validas_perdidas_pct": round(100 * (v - vm) / v, 1) if v else None}


def avaliar_criterio(res, nucleo_sem_filtro_pct, criterio=CRITERIO):
    """res: saída de `linha` da crítica. Retorna (veredito, lista de critérios que falharam)."""
    falhas = []
    if res["nucleo_pct"] is None or res["nucleo_pct"] < criterio["nucleo_min_pct"]:
        falhas.append(f"núcleo entre mantidas {res['nucleo_pct']}% < {criterio['nucleo_min_pct']}%")
    if res["ic95"][0] is None or res["ic95"][0] < criterio["ic_inferior_min_pct"]:
        falhas.append(f"limite inferior do IC95 {res['ic95'][0]}% < {criterio['ic_inferior_min_pct']}%")
    if res["validas_perdidas_pct"] is None or res["validas_perdidas_pct"] > criterio["validas_perdidas_max_pct"]:
        falhas.append(f"válidas perdidas {res['validas_perdidas_pct']}% > {criterio['validas_perdidas_max_pct']}%")
    if res["nucleo_pct"] is not None and res["nucleo_pct"] - nucleo_sem_filtro_pct < criterio["ganho_min_pp"]:
        falhas.append(f"ganho {round(res['nucleo_pct'] - nucleo_sem_filtro_pct, 1)} pp < {criterio['ganho_min_pp']} pp")
    return ("VALIDADO" if not falhas else "NÃO VALIDADO"), falhas


NUCLEO = ("correcao_matematica", "resposta_unica", "dados_suficientes")
# Registrados em ciclo_v8/SPEC.md ANTES de rodar a crítica na folha nova (2026-10-25).
SUSPEITAS = ["Q1006", "Q1007", "Q1008", "Q1013", "Q1014", "Q1017"]


def sortear_revisao(ids, suspeitas, n_controles, seed):
    """Suspeitas + controles sorteados do resto, embaralhados, com ids novos. Os controles medem se o professor muda
    rótulos ao rever qualquer questão (inconsistência), não só as suspeitas."""
    import random
    rng = random.Random(seed)
    resto = sorted(i for i in ids if i not in suspeitas)
    escolhidos = list(suspeitas) + rng.sample(resto, min(n_controles, len(resto)))
    rng.shuffle(escolhidos)
    return {f"R{k + 1:02d}": orig for k, orig in enumerate(escolhidos)}


def aplicar_revisao(folha_linhas, revisao_linhas, mapa):
    """Rótulos do núcleo da revisão substituem os da folha para TODAS as questões revistas (suspeitas e controles),
    para a sensibilidade não escolher só as mudanças favoráveis. Retorna (linhas novas, mudanças)."""
    rev = {mapa[r["id"]]: r for r in revisao_linhas}
    mudou, novas = [], []
    for r in folha_linhas:
        r = dict(r)
        if r["id"] in rev:
            antes = all(r[c].strip() == "s" for c in NUCLEO)
            for c in NUCLEO:
                r[c] = rev[r["id"]][c]
            depois = all(r[c].strip() == "s" for c in NUCLEO)
            if antes != depois:
                mudou.append((r["id"], antes, depois))
        novas.append(r)
    return novas, mudou


def cmd_revisao(a):
    with open(a.folha, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
        campos = list(linhas[0].keys())
    mapa = sortear_revisao([r["id"] for r in linhas], SUSPEITAS, a.controles, a.seed)
    por_id = {r["id"]: r for r in linhas}
    dest = Path(a.saida)
    dest.mkdir(parents=True, exist_ok=True)
    preencher = set(campos[campos.index("correcao_matematica"):])
    with open(dest / "folha_revisao.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for novo, orig in mapa.items():
            w.writerow({c: ("" if c in preencher else por_id[orig][c]) for c in campos} | {"id": novo})
    (dest / "mapa_oculto.json").write_text(json.dumps({"seed": a.seed, "suspeitas": SUSPEITAS, "mapa": mapa},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(mapa)} questões ({len(SUSPEITAS)} suspeitas + {len(mapa) - len(SUSPEITAS)} controles) em {dest}/folha_revisao.csv; "
          "mapa_oculto.json fica com o responsável técnico")


def cmd_novidade(a):
    novas = enunciados(a.folha)
    antigas = {}
    for f in (a.contra or ANTIGAS):
        if Path(f).exists():
            antigas.update({f"{Path(f).parent.name}:{k}": v for k, v in enunciados(f).items()})
    dup = duplicatas(novas, antigas)
    print(f"{len(novas)} questões novas contra {len(antigas)} antigas: {len(dup)} enunciados repetidos")
    if dup:
        print("repetidos:", ", ".join(dup), "(avise o responsável antes de entregar a folha)")
        sys.exit(1)


def cmd_apurar(a):
    folha = a.folha
    if a.revisao:
        # Apuração de SENSIBILIDADE (regra fixada em ciclo_v8/SPEC.md antes da crítica): reportada ao lado da oficial.
        with open(a.folha, encoding="utf-8-sig", newline="") as f:
            linhas = list(csv.DictReader(f))
            campos = list(linhas[0].keys())
        with open(a.revisao, encoding="utf-8-sig", newline="") as f:
            rev = list(csv.DictReader(f))
        if any(not r.get(c, "").strip() for r in rev for c in NUCLEO):
            sys.exit("ABORTADO: a folha de revisão tem células do núcleo em branco.")
        mapa = json.loads(Path(a.mapa).read_text(encoding="utf-8"))["mapa"]
        novas, mudou = aplicar_revisao(linhas, rev, mapa)
        print(f"SENSIBILIDADE: {len(mudou)} rótulos de núcleo mudaram na revisão cega: "
              + ", ".join(f"{i} {'válida' if x else 'inválida'}→{'válida' if y else 'inválida'}"
                          + (" (suspeita)" if i in SUSPEITAS else " (controle)") for i, x, y in mudou))
        SAIDA.mkdir(parents=True, exist_ok=True)
        folha = SAIDA / f"folha_sensibilidade_{a.rotulo}.csv"
        with open(folha, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(novas)
        a.rotulo_saida = f"{a.rotulo}_sensibilidade"
    args = argparse.Namespace(modo="critica", permutar=True, rotulo=a.rotulo)
    itens = ac.itens_de_folha(folha, a.chave)
    with open(folha, encoding="utf-8-sig", newline="") as f:
        em_branco = [r["id"] for r in csv.DictReader(f)
                     if any(not r.get(c, "").strip() for c in ("correcao_matematica", "resposta_unica", "dados_suficientes"))]
    if em_branco:
        sys.exit(f"ABORTADO: {len(em_branco)} questões sem os três critérios do núcleo preenchidos ({', '.join(em_branco[:8])}...). "
                 "Use a folha devolvida pelo professor, completa.")
    crit_p = ac.SAIDA / f"respostas_{ac.sufixo(args)}.jsonl"
    cot_p = ac.SAIDA / f"respostas_{ac.sufixo(argparse.Namespace(modo='cot', permutar=True, rotulo=a.rotulo))}.jsonl"
    crit, motivos = ac.ler_respostas(crit_p, itens, "critica", True)
    faltam = [x["uid"] for x in itens if x["uid"] not in crit]
    if faltam:
        sys.exit(f"ABORTADO: faltam respostas da crítica para {len(faltam)} questões ({', '.join(faltam[:8])}...). "
                 "Termine o `resolver` antes de apurar.")
    cot = ac.ler_respostas(cot_p, itens, "cot", True)[0] if cot_p.exists() else None
    if cot is not None and any(x["uid"] not in cot for x in itens):
        sys.exit("ABORTADO: a rodada da cot está incompleta; termine o `resolver --modo cot` ou apague o arquivo.")
    gab = {x["uid"]: x["gabarito_modelo"] for x in itens}
    c_ok = lambda x: crit.get(x["uid"]) == gab[x["uid"]]
    k_ok = lambda x: cot is not None and cot.get(x["uid"]) == gab[x["uid"]]
    v_ok = lambda x: x.get("status_verificador") == "ok"
    n_val = sum(bool(x["valida_nucleo"]) for x in itens)
    sem_filtro = round(100 * n_val / len(itens), 1)
    quadro = {"sem filtro": linha(itens, lambda x: True), "crítica": linha(itens, c_ok),
              "verificador ok": linha(itens, v_ok), "crítica e verificador ok": linha(itens, lambda x: c_ok(x) and v_ok(x))}
    if cot is not None:
        quadro.update({"cot": linha(itens, k_ok), "cot e crítica": linha(itens, lambda x: k_ok(x) and c_ok(x))})
    veredito, falhas = avaliar_criterio(quadro["crítica"], sem_filtro)
    por_ano = {}
    for ano in sorted({x["ano"] for x in itens}):
        sub = [x for x in itens if x["ano"] == ano]
        por_ano[ano] = {"n": len(sub), "sem_filtro_pct": round(100 * sum(bool(x["valida_nucleo"]) for x in sub) / len(sub), 1),
                        **{k: v for k, v in linha(sub, c_ok).items() if k in ("mantidas", "nucleo")}}
    motivo_final = Counter("uma_correta_outra_letra" if m == "uma_correta" and not c_ok(x) else m
                           for x in itens for m in [motivos[x["uid"]]])
    res = {"n": len(itens), "nucleo_sem_filtro_pct": sem_filtro, "criterio": CRITERIO, "veredito": veredito,
           "criterios_que_falharam": falhas, "quadro": quadro, "crítica_por_ano": por_ano,
           "motivos_da_critica": dict(motivo_final), "referencia_ajuste_v8rftb": "85,2% [73,4; 92,3] (n=78, usado para ajustar o prompt)"}
    print(f"{len(itens)} questões, núcleo sem filtro {sem_filtro}%")
    for nome, q in quadro.items():
        print(f"  {nome:26s} mantidas {q['mantidas']:3d} ({q['mantidas_pct']:5.1f}%)  núcleo {q['nucleo']:>7s} = {q['nucleo_pct']}% "
              f"IC95 {q['ic95']}  válidas perdidas {q['validas_perdidas']} ({q['validas_perdidas_pct']}%)")
    print("  por ano (crítica):", {k: f"{v['nucleo']} de {v['mantidas']} (sem filtro {v['sem_filtro_pct']}%, n={v['n']})" for k, v in por_ano.items()})
    print("  motivos:", dict(motivo_final))
    print(f"\nVEREDITO: {veredito}" + ("" if not falhas else "\n  - " + "\n  - ".join(falhas)))
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / f"apuracao_{getattr(a, 'rotulo_saida', a.rotulo)}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("novidade")
    n.add_argument("--folha", required=True)
    n.add_argument("--contra", nargs="*", help="folhas antigas (padrão: v8rftb de 78 e as 222 do v7b)")
    s = sub.add_parser("apurar")
    s.add_argument("--folha", required=True, help="folha PREENCHIDA pelo professor")
    s.add_argument("--chave", required=True)
    s.add_argument("--rotulo", required=True, help="o mesmo --rotulo usado no `resolver`")
    s.add_argument("--revisao", help="folha_revisao preenchida: roda a apuração de SENSIBILIDADE")
    s.add_argument("--mapa", help="mapa_oculto.json da revisão (obrigatório com --revisao)")
    r = sub.add_parser("revisao")
    r.add_argument("--folha", required=True, help="folha PREENCHIDA pelo professor")
    r.add_argument("--saida", required=True)
    r.add_argument("--controles", type=int, default=10)
    r.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    if a.cmd == "apurar" and a.revisao and not a.mapa:
        ap.error("--revisao exige --mapa")
    {"novidade": cmd_novidade, "apurar": cmd_apurar, "revisao": cmd_revisao}[a.cmd](a)


if __name__ == "__main__":
    main()
