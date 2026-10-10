"""Consolidação da base: decisões da resolução cega, remoções e balanceamento (sem rede)."""
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import consolidar_base as cb  # noqa: E402

Q = {"enunciado": "Pedro tinha 25 figurinhas e comprou 17. Quantas tem agora?",
     "alternativas": {"A": "38", "B": "42", "C": "43", "D": "44", "E": "45"},
     "resolucao_passo_a_passo": "Correto - 25 + 17 = 42", "resposta_correta": "B", "difficulty": "EASY"}
RES_BOA = "Pedro tinha 25 figurinhas e comprou mais 17. Somamos: 25 + 17 = 42. Portanto, ele tem 42 figurinhas."


class Falso:
    """Cliente que responde a pergunta de verdade: acha a linha '42' e devolve a letra exibida."""

    def __init__(self, valor="42", resolucao=RES_BOA, cego=False):
        self.valor, self.resolucao, self.cego, self.n = valor, resolucao, cego, 0

    def chat_completion(self, messages, max_tokens, temperature):
        self.n += 1
        letra = "X"
        for linha in messages[-1]["content"].splitlines():
            if linha[3:].strip() == self.valor:
                letra = linha[0]
        if self.cego:
            letra = "A" if self.n % 2 else "B"  # discorda de si mesmo
        txt = json.dumps({"resposta": letra, "resolucao": self.resolucao})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=txt))])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = cb.CACHE
        cb.CACHE = Path(self.tmp.name) / "c.jsonl"

    def tearDown(self):
        cb.CACHE = self.old
        self.tmp.cleanup()

    def ch(self, cliente, teto=50):
        return cb.Chamador("m", cliente, teto)


class TestResolverItem(Base):
    def test_mantem_gabarito_e_troca_a_resolucao(self):
        d = cb.resolver_item(Q, self.ch(Falso()), 0)
        self.assertEqual((d["decisao"], d["gabarito"]), ("manter", "B"))
        self.assertIn("25 + 17 = 42", d["resolucao"])

    def test_discorda_sem_confirmacao_aritmetica_remove(self):
        # modelo acha "44" (D) e a resolução não prova: o verificador reprova ou não confirma -> remove
        d = cb.resolver_item(Q, self.ch(Falso(valor="44", resolucao="Acho que a resposta é 44 porque sim, pela lógica do problema todo.")), 0)
        self.assertEqual(d["decisao"], "remover")

    def test_discorda_com_confirmacao_aritmetica_troca(self):
        q = dict(Q, resposta_correta="D")  # gabarito errado: 44
        d = cb.resolver_item(q, self.ch(Falso()), 0)
        self.assertEqual((d["decisao"], d["gabarito"], d["gabarito_antigo"]), ("trocar_gabarito", "B", "D"))

    def test_passes_que_discordam_removem(self):
        d = cb.resolver_item(Q, self.ch(Falso(cego=True)), 0)
        self.assertEqual(d["decisao"], "remover")

    def test_resolucao_invalida_mantem_a_antiga_se_gabarito_confirmado(self):
        d = cb.resolver_item(Q, self.ch(Falso(resolucao="A resposta é a alternativa B porque 42.")), 0)
        self.assertEqual(d["decisao"], "manter_resolucao_antiga")

    def test_validacoes_da_resolucao(self):
        self.assertEqual(cb._resolucao_valida(Q, "B", "curta")[1], "resolucao_curta")
        self.assertEqual(cb._resolucao_valida(Q, "B", "Somamos 25 + 17 = 42. Logo é a alternativa B, com certeza total.")[1], "cita_letra_de_alternativa")
        self.assertEqual(cb._resolucao_valida(Q, "B", "PEDRO TINHA 25 FIGURINHAS E COMPROU 17, SOMANDO 25 + 17 = 42 FIGURINHAS.")[1], "caixa_alta")
        self.assertTrue(cb._resolucao_valida(Q, "B", RES_BOA)[0])


class TestTeto(Base):
    def test_teto_esgotado_aborta_em_vez_de_remover(self):
        ch = self.ch(Falso(), teto=1)
        cb.resolver_item(Q, ch, 0)
        with self.assertRaises(SystemExit):
            ch.checar()


class TestMontar(Base):
    def _arquivo(self, itens):
        p = Path(self.tmp.name) / "in.jsonl"
        linhas = []
        for cod, q in itens:
            linhas.append(json.dumps({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                                   {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
                                      "meta": {"codigo_item": cod}}, ensure_ascii=False))
        p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        return p

    def test_remove_aplica_resolucao_e_balanceia(self):
        itens = []
        for k in range(20):
            q = json.loads(json.dumps(Q))
            q["enunciado"] = f"Pedro tinha {20 + k} figurinhas e comprou 17. Quantas tem agora?"
            q["alternativas"] = {"A": str(20 + k + 17), "B": str(20 + k + 18), "C": str(20 + k + 19), "D": str(20 + k + 20), "E": str(20 + k + 21)}
            q["resposta_correta"] = "A"
            q["resolucao_passo_a_passo"] = f"{20 + k} + 17 = {20 + k + 17}"
            itens.append((f"DIST-{k}", q))
        p = self._arquivo(itens)
        figura = {"DIST-0": {"resolvivel": False, "falta": "figura"}, "DIST-1": {"resolvivel": True, "falta": ""}}
        resolver = {"DIST-2": {"decisao": "remover", "motivo": "passes_discordam_ou_sem_resposta", "passes": []},
                    "DIST-3": {"decisao": "manter", "gabarito": "A", "resolucao": RES_BOA, "verificador": True, "passes": []}}
        out, log = Path(self.tmp.name) / "out.jsonl", Path(self.tmp.name) / "log.jsonl"
        r = cb.montar(p, out, figura, resolver, propostas=None, log=log)
        self.assertEqual((r["exemplos_entrada"], r["exemplos_saida"]), (20, 18))
        linhas = [json.loads(l) for l in open(out, encoding="utf-8")]
        cods = [l["meta"]["codigo_item"] for l in linhas]
        self.assertNotIn("DIST-0", cods)
        self.assertNotIn("DIST-2", cods)
        qs = {l["meta"]["codigo_item"]: json.loads(l["messages"][2]["content"])["questoes"][0] for l in linhas}
        self.assertEqual(qs["DIST-3"]["resolucao_passo_a_passo"], RES_BOA)
        for q in qs.values():  # o texto da correta viaja junto com a letra
            self.assertTrue(q["alternativas"][q["resposta_correta"]].isdigit())
        letras = Counter(q["resposta_correta"] for q in qs.values())
        self.assertLessEqual(max(letras.values()) - min(letras.get(L, 0) for L in "ABCDE"), 2)

    def test_recusa_arquivos_protegidos(self):
        p = self._arquivo([("DIST-0", Q)])
        for nome in ("train_curado.jsonl", "train.jsonl", "val_frozen_v1.jsonl"):
            with self.assertRaises(SystemExit):
                cb.montar(p, Path(self.tmp.name) / nome, {}, {}, propostas=None, log=Path(self.tmp.name) / "l")


if __name__ == "__main__":
    unittest.main()
