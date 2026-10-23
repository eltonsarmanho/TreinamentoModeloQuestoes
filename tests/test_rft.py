import json, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rft_filtrar as rf  # noqa: E402


def amostra(pid, enun="Quanto é 15 + 20?", status="ok", alts=None):
    q = {"enunciado": enun, "alternativas": alts or {"A": "30", "B": "25", "C": "35", "D": "45", "E": "Nenhuma"},
         "resolucao_passo_a_passo": "Somando as quantidades do enunciado: 15 + 20 = 35.", "resposta_correta": "C", "difficulty": "EASY"}
    return {"pid": pid, "k": 0, "ano": "2º", "habilidade": "H06", "dificuldade": "Fácil", "prompt": "p",
            "status": status, "obj": {"questoes": [q]}}


class T(unittest.TestCase):
    def test_camada1(self):
        ok, rej = rf.camada1([amostra(1), amostra(2), amostra(3, enun="Outra?", status="falha"),
                              amostra(4, enun="Z?", alts={"A": "1", "B": "1", "C": "2", "D": "3", "E": "4"}),
                              {"pid": 5, "k": 0, "obj": None, "status": "falha"}])
        self.assertEqual([s["pid"] for s in ok], [1])
        self.assertEqual(sorted(m for _, m in rej), ["alternativas_repetidas", "duplicata", "sem_questao", "status_falha"])

    def test_aceita_obj_direto_e_envelope(self):
        a = amostra(1); direto = dict(a, obj=a["obj"]["questoes"][0], pid=2, **{})
        direto["obj"] = dict(direto["obj"], enunciado="Outra?")
        ok, _ = rf.camada1([a, direto])
        self.assertEqual(len(ok), 2)

    def test_extras_saem_com_letra_balanceada(self):
        import tempfile, collections
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); base = d / "b.jsonl"
            base.write_text("".join(json.dumps({"messages": [], "meta": {}}) + "\n" for _ in range(400)))
            ac = []
            for i in range(60):  # todos com gabarito C antes da permutação
                a = amostra(i, enun=f"Quanto é {i} + 20?"); q = a["obj"]["questoes"][0]
                q["alternativas"] = {"A": "30", "B": "25", "C": str(i + 20), "D": "45", "E": "50"}
                q["resolucao_passo_a_passo"] = f"Somando as quantidades do enunciado: {i} + 20 = {i + 20}."
                ac.append({**a, "q": q})
            ne, _, _ = rf.montar(ac, [], base, d / "e.jsonl", d / "t.jsonl", d / "p.jsonl")
            self.assertEqual(ne, 60)
            letras = collections.Counter()
            for l in (d / "e.jsonl").read_text().splitlines():
                q = json.loads(json.loads(l)["messages"][2]["content"])["questoes"][0]
                letras[q["resposta_correta"]] += 1
                self.assertEqual(q["alternativas"][q["resposta_correta"]], q["enunciado"].split()[2].rstrip("?") and str(int(q["enunciado"].split()[2]) + 20))
            self.assertLessEqual(max(letras.values()), 60 * 0.3)

    def test_norm_ignora_acento_e_caixa(self):
        self.assertEqual(rf.norm("Quanto É Ação?"), rf.norm("quanto e acao"))

    def test_montar_ordem_v8_e_teto(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base = d / "b.jsonl"
            base.write_text("".join(json.dumps({"messages": [], "meta": {}}) + "\n" for _ in range(10)))
            ac = [{**amostra(i, enun=f"Q{i}?"), "q": amostra(i, enun=f"Q{i}?")["obj"]["questoes"][0]} for i in range(6)]
            ne, npar, teto = rf.montar(ac, [], base, d / "e.jsonl", d / "t.jsonl", d / "p.jsonl")
            self.assertEqual(teto, 3)
            self.assertEqual(ne, 3)
            q = json.loads(json.loads((d / "e.jsonl").read_text().splitlines()[0])["messages"][2]["content"])["questoes"][0]
            self.assertEqual(list(q)[:3], ["enunciado", "resolucao_passo_a_passo", "alternativas"])
            self.assertEqual(len((d / "t.jsonl").read_text().splitlines()), 13)


if __name__ == "__main__":
    unittest.main()


class TAval(unittest.TestCase):
    def test_sem_dedupe_e_mcnemar(self):
        import avaliar_jev as av
        ok, _ = rf.camada1_sem_dedupe([amostra(1), amostra(2)])  # mesmo enunciado: dedupe NÃO se aplica
        self.assertEqual(len(ok), 2)
        self.assertEqual(av.mcnemar_exato(0, 0), 1.0)
        self.assertLess(av.mcnemar_exato(0, 10), 0.01)
