"""Ciclo v8 (opção A): resolver ANTES de escrever as alternativas. Determinístico, sem API.

Problema medido: o professor reprova no 4B, sobretudo, itens cuja resolução calcula o valor certo mas
o valor não está entre as cinco alternativas (ou a letra marcada não é a dele). Na ordem de emissão
atual (enunciado, alternativas, resolução, letra, dificuldade) as cinco opções são escritas antes da
conta existir. Aqui o alvo de treino passa a emitir enunciado, resolução, alternativas, letra,
dificuldade: o valor já está calculado quando as alternativas são escritas.

O contrato do app não muda: mesmas chaves, mesma estrutura, 5 alternativas A-E; o app lê o JSON por
nome de chave (ver schema_utils). Muda a gramática (grammars/questao_v8.gbnf) e o alvo de treino.

Transformações (todas em texto, nenhuma toca enunciado, alternativas nem gabarito):
  1. ordem das chaves de cada questão;
  2. exemplos multi-questão viram um exemplo por questão ("Gere 1 questão(ões)"), o formato que o app
     usa (gerar_lote.py chama uma vez por questão);
  3. frases da resolução que apontam para as alternativas ("a alternativa correta é a C", "Comparando
     com as opções apresentadas...") saem, porque na nova ordem as alternativas ainda não existem.
     Uma frase só sai se for padrão inequívoco de referência (ver REFERENCIA); "3 opções de camisa"
     num problema de contagem fica.

Trava: depois da limpeza a questão passa de novo por schema_utils.check_consistency_detalhado. O
veredito não pode mudar (True continua True, None continua None ou vira True, False continua False).
Questão que mudaria de veredito, ou cuja resolução ficaria com menos de MIN_RESOLUCAO caracteres,
tenta a limpeza mínima (só frases finais); se ainda assim falhar, sai do corpus e vai para o relatório.

    venv/bin/python src/construir_v8.py            # escreve data/train_curado_v8.jsonl e o holdout de loss

Nunca escreve em train_curado.jsonl, train.jsonl nem val*.jsonl. Entradas lidas só para leitura.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import schema_utils as su  # noqa: E402

BASE_PADRAO = ROOT / "data" / "train_curado_v7b_fix1.jsonl"
SAIDA_PADRAO = ROOT / "data" / "train_curado_v8.jsonl"
VAL_PADRAO = ROOT / "data" / "val_frozen_v1.jsonl"
HOLDOUT_PADRAO = ROOT / "data" / "holdout_loss_v8.jsonl"
RELATORIO_PADRAO = ROOT / "outputs" / "melhoria_v8" / "relatorio_v8.json"

ORDEM_V8 = ("enunciado", "resolucao_passo_a_passo", "alternativas", "resposta_correta", "difficulty")
MIN_RESOLUCAO = 30

# Referência INEQUÍVOCA às alternativas. "opção/opções" sozinhas ficam de fora: em problemas de
# contagem ("3 opções de camisa") são conteúdo da matemática, não referência ao gabarito.
REFERENCIA = re.compile(
    r"\balternativas?\b"
    r"|\bop[cç][aã]o\s+(?:correta|certa|[A-E]\b)"
    r"|\bop[cç][õo]es\s+(?:apresentadas|dadas|acima|abaixo|de resposta|dispon[ií]veis|a seguir)"
    r"|\b(?:entre|nas|das|com as|comparando com as|analisando as|verificando as)\s+op[cç][õo]es\b"
    r"|\bletra\s+[A-E]\b"
    r"|\(\s*[A-E]\s*\)"
    r"|\bgabarito\b",
    re.I,
)
MARCADOR = re.compile(r"(?<![\w(])([A-E])\)")  # "A) ... B) ..." listando as opções
FRONTEIRA = re.compile(r"(?<=[.!?])(\s+)(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“(])|(\n+)")
PROMPT_N = re.compile(r"Gere\s+\d+\s+questão\(ões\)")


def _frases(texto):
    """Quebra em (frase, separador que vem depois). Reconstrói o texto exato se nada sair."""
    partes, ini = [], 0
    for m in FRONTEIRA.finditer(texto):
        partes.append((texto[ini:m.start()], m.group(0)))
        ini = m.end()
    partes.append((texto[ini:], ""))
    return partes


def e_referencia(frase):
    return bool(REFERENCIA.search(frase)) or len(set(MARCADOR.findall(frase))) >= 2


def limpar_resolucao(res, so_finais=False):
    """Devolve (resolucao_nova, frases_removidas). `so_finais`: só remove o bloco final de referências."""
    partes = _frases(res.strip())
    marcadas = [e_referencia(f) for f, _ in partes]
    if so_finais:
        k = len(partes)
        while k > 0 and marcadas[k - 1]:
            k -= 1
        manter = [i < k for i in range(len(partes))]
    else:
        manter = [not m for m in marcadas]
    saida, removidas, ultimo_sep = [], [], " "
    for (frase, sep), ficar in zip(partes, manter):
        if ficar:
            if saida:
                saida.append(ultimo_sep)
            saida.append(frase)
            ultimo_sep = sep if sep else " "
        else:
            removidas.append(frase)
    return "".join(saida).strip(), removidas


def ordenar_v8(q):
    faltam = [k for k in ORDEM_V8 if k not in q]
    if faltam:
        raise KeyError(f"questão sem as chaves {faltam}")
    extra = [k for k in q if k not in ORDEM_V8]
    if extra:
        raise KeyError(f"chaves fora do contrato: {extra}")
    return {k: q[k] for k in ORDEM_V8}


def _veredito(q):
    ok, _sug, motivo = su.check_consistency_detalhado(q)
    return ok, motivo


def _compativel(antes, depois):
    """True continua True; None continua None ou vira True; False continua False."""
    if antes is True:
        return depois is True
    if antes is None:
        return depois in (None, True)
    return depois is False


def transformar_questao(q, preservar=False):
    """(questao_nova | None, info). None = questão fora do corpus (motivo em info['descartada']).

    `preservar`: item REAL (banco SAEB / lote L2). Remover item real exige decisão humana
    (ciclo_v7/SPEC.md); se a trava reprova a limpeza, o item fica, só reordenado, com a resolução
    original (modo "mantida_sem_limpeza")."""
    info = {"frases_removidas": 0, "modo": "sem_referencia"}
    antes, motivo_antes = _veredito(q)
    nova = dict(q)
    res = q["resolucao_passo_a_passo"]
    if any(e_referencia(f) for f, _ in _frases(res.strip())):
        for modo, so_finais in (("completa", False), ("so_finais", True)):
            cand, removidas = limpar_resolucao(res, so_finais=so_finais)
            teste = dict(q, resolucao_passo_a_passo=cand)
            depois, _ = _veredito(teste)
            if len(cand) >= MIN_RESOLUCAO and _compativel(antes, depois):
                nova, info = teste, {"frases_removidas": len(removidas), "modo": modo}
                break
        else:
            if preservar:
                return ordenar_v8(q), {"frases_removidas": 0, "modo": "mantida_sem_limpeza",
                                       "veredito_antes": antes}
            info = {"frases_removidas": 0, "modo": "reprovada_na_trava",
                    "descartada": f"veredito {antes}/{motivo_antes} mudaria ou resolução curta demais"}
            return None, info
    info["veredito_antes"] = antes
    return ordenar_v8(nova), info


def serializar(questoes):
    return json.dumps({"questoes": questoes}, ensure_ascii=False)


def transformar_exemplo(ex):
    """Um exemplo -> lista de exemplos v8 (um por questão aproveitada) e lista de infos."""
    msgs = ex["messages"]
    questoes = json.loads(msgs[2]["content"])["questoes"]
    multi = len(questoes) > 1
    saida, infos = [], []
    for i, q in enumerate(questoes):
        codigo = ex.get("meta", {}).get("codigo_item")
        nova, info = transformar_questao(q, preservar=eh_real(codigo))
        info["codigo_item"] = codigo
        info["dividido"] = multi
        infos.append(info)
        if nova is None:
            continue
        user = PROMPT_N.sub("Gere 1 questão(ões)", msgs[1]["content"]) if multi else msgs[1]["content"]
        meta = dict(ex.get("meta", {}))
        if multi:
            meta["codigo_item"] = f"{meta.get('codigo_item', 'x')}-q{i + 1}"
        meta["v8"] = {"fonte": "train_curado_v7b_fix1", "modo_limpeza": info["modo"],
                      "frases_removidas": info["frases_removidas"], "dividido": multi}
        saida.append({"messages": [msgs[0], {"role": "user", "content": user},
                                   {"role": "assistant", "content": serializar([nova])}],
                      "meta": meta})
    return saida, infos


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ler(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def escrever(p, linhas):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas), encoding="utf-8")


def eh_real(codigo):
    return origem(codigo) in ("banco", "L2")


def origem(codigo):
    c = codigo or ""
    for pref in ("INJ", "DIST", "REFORCO", "SINT"):
        if c.startswith(pref):
            return pref
    return "L2" if "-L2-" in c else "banco"


def construir(base, saida, val, holdout, relatorio):
    for alvo in (saida, holdout):
        nome = Path(alvo).name
        if nome in ("train_curado.jsonl", "train.jsonl") or nome.startswith("val"):
            raise SystemExit(f"recuso escrever em {nome}: arquivo protegido")
    exemplos = ler(base)
    novos, infos = [], []
    for ex in exemplos:
        s, i = transformar_exemplo(ex)
        novos.extend(s)
        infos.extend(i)
    modos = Counter(i["modo"] for i in infos)
    descartadas = [i for i in infos if "descartada" in i]
    por_origem = Counter(origem(i["codigo_item"]) for i in infos if i["modo"] in ("completa", "so_finais"))
    descartadas_por_origem = Counter(origem(i["codigo_item"]) for i in infos if "descartada" in i)
    mantidas = [i["codigo_item"] for i in infos if i["modo"] == "mantida_sem_limpeza"]
    verdicts = Counter(str(i.get("veredito_antes")) for i in infos if "descartada" not in i)
    escrever(saida, novos)

    # holdout de loss: val congelado na MESMA ordem, sem trava (só mede loss; o gate usa só os prompts)
    hold = []
    for ex in ler(val):
        msgs = ex["messages"]
        qs = json.loads(msgs[2]["content"])["questoes"]
        qn = []
        for q in qs:
            nova, _ = transformar_questao(q, preservar=True)
            qn.append(nova if nova is not None else ordenar_v8(q))
        hold.append({"messages": [msgs[0], msgs[1], {"role": "assistant", "content": serializar(qn)}],
                     "meta": ex.get("meta", {})})
    escrever(holdout, hold)

    rel = {
        "entrada": {"arquivo": _rel(base), "sha256": sha256(base),
                    "exemplos": len(exemplos), "questoes": len(infos)},
        "saida": {"arquivo": _rel(saida), "sha256": sha256(saida),
                  "exemplos": len(novos), "questoes": len(novos)},
        "holdout_loss": {"arquivo": _rel(holdout), "sha256": sha256(holdout),
                         "exemplos": len(hold), "derivado_de": _rel(val),
                         "sha256_fonte": sha256(val)},
        "multi_questao_divididos": sum(1 for i in infos if i["dividido"]),
        "limpeza": dict(modos),
        "limpeza_por_origem": dict(por_origem),
        "veredito_verificador_antes": dict(verdicts),
        "descartadas": len(descartadas),
        "descartadas_por_origem": dict(descartadas_por_origem),
        "reais_mantidos_sem_limpeza": mantidas,
        "descartadas_detalhe": descartadas[:50],
    }
    Path(relatorio).parent.mkdir(parents=True, exist_ok=True)
    Path(relatorio).write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(BASE_PADRAO))
    ap.add_argument("--saida", default=str(SAIDA_PADRAO))
    ap.add_argument("--val", default=str(VAL_PADRAO))
    ap.add_argument("--holdout", default=str(HOLDOUT_PADRAO))
    ap.add_argument("--relatorio", default=str(RELATORIO_PADRAO))
    a = ap.parse_args()
    rel = construir(Path(a.base), Path(a.saida), Path(a.val), Path(a.holdout), Path(a.relatorio))
    print(json.dumps({k: v for k, v in rel.items() if k != "descartadas_detalhe"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
