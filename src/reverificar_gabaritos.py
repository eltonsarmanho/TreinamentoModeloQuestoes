"""Resolução cega paga dos gabaritos FRACOS (Maritaca), com teto duro de chamadas.

Entrada: outputs/relatorios/AUDITORIA_GABARITOS_v3.jsonl (src/auditar_gabaritos_base.py);
só os itens de classe FRACO (juízes "baixa"/"media" ou nunca julgados) são resolvidos.

Como funciona: o modelo recebe SÓ o enunciado e as alternativas — nunca o gabarito, a
resolução do autor nem a dificuldade. Cada item é resolvido em 2 passes com as alternativas
em permutações diferentes (anula viés de posição) e os dois passes têm de apontar a MESMA
alternativa de conteúdo. Comparação com o gabarito é feita aqui, em código.

Classes de saída por item:
  confirmado_cego   — os 2 passes concordam entre si e com o gabarito.
  proposta_trocar   — os 2 passes concordam entre si e DISCORDAM do gabarito. É só uma
                      PROPOSTA: nada é alterado; item real do banco e decisão humana
                      (Doc/decisoes_humanas.json) têm precedência e exigem revisão humana.
  inconclusivo      — os passes discordam entre si, ou não responderam: fica para revisão.

Não escreve na base. Saída: outputs/relatorios/PROPOSTAS_CORRECAO_GABARITO.jsonl.
Uso: venv/bin/python src/reverificar_gabaritos.py --real --modelo sabia-4-thinking --max-chamadas 200
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

AUDITORIA = ROOT / "outputs" / "relatorios" / "AUDITORIA_GABARITOS_v3.jsonl"
SAIDA = ROOT / "outputs" / "relatorios" / "PROPOSTAS_CORRECAO_GABARITO.jsonl"
CACHE = ROOT / "outputs" / "agentes" / "resolucao_cega_fracos_cache.jsonl"

SISTEMA = ("Você é um professor de matemática do ensino básico brasileiro. Resolva a questão do zero, "
           "com rigor, e escolha a alternativa correta. Se nenhuma alternativa estiver correta, ou mais de "
           'uma estiver, responda "X". Responda APENAS com JSON: {"resposta": "A|B|C|D|E|X", "motivo": "<uma frase>"}.')


def montar_pergunta(q, permutacao):
    """Alternativas reordenadas: permutacao = lista das letras ORIGINAIS na ordem em que aparecem."""
    letras = "ABCDE"
    alts = [f"{letras[i]}) {q['alternativas'][orig]}" for i, orig in enumerate(permutacao)]
    return f"{q['enunciado']}\n\n" + "\n".join(alts)


def permutacoes(q, seed):
    orig = [L for L in "ABCDE" if L in q["alternativas"]]
    rnd = random.Random(seed)
    p1 = orig[:]
    p2 = orig[::-1]
    if p2 == p1:
        rnd.shuffle(p2)
    return [p1, p2]


def parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        r = str(json.loads(m.group(0)).get("resposta", "")).strip().upper()
    except json.JSONDecodeError:
        return None
    return r if r in list("ABCDEX") else None


class Resolvedor:
    def __init__(self, modelo, cliente, max_chamadas):
        self.modelo, self.cliente, self.max_chamadas, self.chamadas = modelo, cliente, max_chamadas, 0
        self.cache = {}
        if CACHE.exists():
            for l in CACHE.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    self.cache[r["chave"]] = r["texto"]

    def _chamar(self, pergunta):
        chave = hashlib.sha256((self.modelo + SISTEMA + pergunta).encode("utf-8")).hexdigest()
        if chave in self.cache:
            return self.cache[chave]
        if self.chamadas >= self.max_chamadas:
            return None
        self.chamadas += 1
        for t in range(3):
            try:
                r = self.cliente.chat_completion([{"role": "system", "content": SISTEMA},
                                                  {"role": "user", "content": pergunta}], max_tokens=6000, temperature=0.0)
                txt = r.choices[0].message.content
                self.cache[chave] = txt
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"chave": chave, "modelo": self.modelo, "texto": txt}, ensure_ascii=False) + "\n")
                return txt
            except Exception:  # noqa: BLE001 — rede/cota: tenta de novo; a chave nunca é impressa
                time.sleep(3 * (t + 1))
        return None

    def resolver(self, q, seed):
        """(classe, respostas_originais). respostas em letra ORIGINAL ('X' = nenhuma/mais de uma)."""
        resp = []
        for perm in permutacoes(q, seed):
            r = parse(self._chamar(montar_pergunta(q, perm)))
            resp.append(None if r is None else ("X" if r == "X" else perm["ABCDE".index(r)] if "ABCDE".index(r) < len(perm) else None))
        gab = q["resposta_correta"]
        if None in resp or resp[0] != resp[1] or resp[0] == "X":
            return "inconclusivo", resp
        return ("confirmado_cego" if resp[0] == gab else "proposta_trocar"), resp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v4.jsonl"),
                    help="texto lido para resolver (v4 = já padronizado; o gabarito é o mesmo da v3)")
    ap.add_argument("--classes", default="FRACO")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--modelo", default="sabia-4-thinking")
    ap.add_argument("--max-chamadas", type=int, default=0)
    args = ap.parse_args()
    classes = set(args.classes.split(","))
    alvo = [json.loads(l) for l in open(AUDITORIA, encoding="utf-8") if l.strip()]
    alvo = [x for x in alvo if x["classe"] in classes]
    base = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    print(f"{len(alvo)} itens a resolver; chamadas previstas: {2 * len(alvo)} (cache poupa as repetidas)")
    if not args.real:
        print("Simulação: nenhuma chamada feita. Use --real --max-chamadas N.")
        return
    if args.max_chamadas <= 0:
        sys.exit("ABORTADO: --real exige --max-chamadas N > 0 (teto duro).")
    import distill_teacher as dt  # carrega o .env; a chave nunca é impressa
    rv = Resolvedor(args.modelo, dt.MaritacaClient(args.modelo, timeout=300), args.max_chamadas)
    saidas, cont = [], Counter()
    for k, x in enumerate(alvo):
        q = json.loads(base[x["linha"]]["messages"][2]["content"])["questoes"][x["questao"]]
        assert q["resposta_correta"] == x["gabarito"], "base e auditoria desalinhadas"
        classe, resp = rv.resolver(q, seed=k)
        cont[classe] += 1
        saidas.append({"codigo_item": x["codigo_item"], "ano": x["ano"], "habilidade": x["habilidade"],
                       "unidade": x["unidade"], "origem": x["origem"], "gabarito_atual": q["resposta_correta"],
                       "passes": resp, "classe": classe, "evidencias_auditoria": x["evidencias"]})
        print(f"[{k + 1}/{len(alvo)}] {x['codigo_item']} {classe} gab={q['resposta_correta']} passes={resp}", flush=True)
    SAIDA.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in saidas), encoding="utf-8")
    print(dict(cont), "| chamadas:", rv.chamadas)


if __name__ == "__main__":
    main()
