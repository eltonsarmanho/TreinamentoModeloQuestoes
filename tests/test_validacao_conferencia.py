import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import validacao_conferencia as vc


def _it(uid, valido, ok=False):
    return {"uid": uid, "valida_nucleo": valido, "status_verificador": "ok" if ok else "nao_verificavel"}


class TestValidacaoConferencia(unittest.TestCase):
    def test_duplicata_ignora_acento_caixa_e_pontuacao(self):
        novas = {"Q1": vc.normalizar("Quantos AZULEJOS há, afinal?"), "Q2": vc.normalizar("outra coisa")}
        self.assertEqual(vc.duplicatas(novas, {"x:Q9": vc.normalizar("Quantos azulejos havia afinal")}), [])
        self.assertEqual(vc.duplicatas(novas, {"x:Q9": vc.normalizar("quantos azulejos ha afinal")}), ["Q1"])

    def test_linha_conta_mantidas_e_validas_perdidas(self):
        itens = [_it("a", True), _it("b", True), _it("c", False), _it("d", False)]
        r = vc.linha(itens, lambda x: x["uid"] in ("a", "c"))
        self.assertEqual((r["mantidas"], r["nucleo"], r["validas_perdidas"]), (2, "1/2", "1/2"))
        self.assertEqual(r["nucleo_pct"], 50.0)

    def test_criterio_aprova_so_com_todos(self):
        bom = {"nucleo_pct": 85.0, "ic95": [76.0, 91.0], "validas_perdidas_pct": 10.0}
        self.assertEqual(vc.avaliar_criterio(bom, 65.0), ("VALIDADO", []))
        for mudanca, trecho in [({"nucleo_pct": 79.0}, "núcleo"), ({"ic95": [69.0, 90.0]}, "IC95"),
                                ({"validas_perdidas_pct": 16.0}, "perdidas")]:
            v, falhas = vc.avaliar_criterio({**bom, **mudanca}, 65.0)
            self.assertEqual(v, "NÃO VALIDADO")
            self.assertTrue(any(trecho in f for f in falhas), falhas)
        v, falhas = vc.avaliar_criterio(bom, 78.0)  # sem ganho de 10 pp sobre a folha sem filtro
        self.assertEqual(v, "NÃO VALIDADO")
        self.assertTrue(any("ganho" in f for f in falhas))

    def test_criterio_sem_mantidas_nao_aprova(self):
        v, _ = vc.avaliar_criterio({"nucleo_pct": None, "ic95": [None, None], "validas_perdidas_pct": 100.0}, 65.0)
        self.assertEqual(v, "NÃO VALIDADO")

    def test_revisao_inclui_suspeitas_e_controles_com_ids_novos(self):
        ids = [f"Q{i}" for i in range(1, 31)]
        m = vc.sortear_revisao(ids, ["Q3", "Q7"], 5, 1)
        self.assertEqual(len(m), 7)
        self.assertTrue({"Q3", "Q7"} <= set(m.values()))
        self.assertTrue(all(k.startswith("R") for k in m))
        self.assertEqual(m, vc.sortear_revisao(ids, ["Q3", "Q7"], 5, 1))  # determinista

    def test_sensibilidade_aplica_todas_as_revistas(self):
        s = {c: "s" for c in vc.NUCLEO}
        folha = [{"id": "Q1", **s}, {"id": "Q2", **s}, {"id": "Q3", **s}]
        rev = [{"id": "R01", **s, "resposta_unica": "n"}, {"id": "R02", **s, "dados_suficientes": "n"}]
        novas, mudou = vc.aplicar_revisao(folha, rev, {"R01": "Q1", "R02": "Q3"})
        self.assertEqual(mudou, [("Q1", True, False), ("Q3", True, False)])  # controle Q3 também muda
        self.assertEqual(novas[1], folha[1])


if __name__ == "__main__":
    unittest.main()
