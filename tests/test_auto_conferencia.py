import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import auto_conferencia as ac


def _item(uid, gab, valido, status="nao_verificavel"):
    return {"uid": uid, "sig": uid, "gabarito_modelo": gab, "valida_nucleo": valido, "status_verificador": status,
            "enunciado": "2 + 2?", "alternativas": dict(zip("ABCDE", ["1", "2", "3", "4", "5"]))}


class TestAutoConferencia(unittest.TestCase):
    def test_permutacao_volta_para_a_letra_original(self):
        it = _item("u1", "D", True)
        ordem = ac.ordem_alternativas("u1", True)
        self.assertEqual(sorted(ordem), list("ABCDE"))
        for i, orig in enumerate(ordem):
            self.assertEqual(ac.letra_original("ABCDE"[i], "u1", True), orig)
        self.assertEqual(ac.letra_original("C", "u1", False), "C")
        self.assertEqual(ac.ordem_alternativas("u1", True), ordem)  # determinista
        # o prompt exibe o texto da alternativa original certa na posição sorteada
        pos = ordem.index("D")
        self.assertIn(f"{'ABCDE'[pos]}) 4", ac.montar_prompt(it, "direto", True))

    def test_nenhuma_das_anteriores_fica_por_ultimo(self):
        it = _item("u2", "A", True)
        it["alternativas"]["C"] = "Nenhuma das alternativas anteriores"
        ordem = ac.ordem_alternativas("u2", True, it["alternativas"])
        self.assertEqual(ordem[-1], "C")
        self.assertEqual(sorted(ordem), list("ABCDE"))
        self.assertEqual(ac.letra_original("E", "u2", True, it["alternativas"]), "C")

    def test_extrair_letra(self):
        self.assertEqual(ac.extrair_letra(" b) 5", "direto"), "B")
        self.assertIsNone(ac.extrair_letra("não sei", "direto"))
        self.assertEqual(ac.extrair_letra("passos... Resposta: **C**", "cot"), "C")
        self.assertEqual(ac.extrair_letra("Resposta: A\nmas Resposta: E", "cot"), "E")
        self.assertEqual(ac.extrair_letra("**Resposta:** C", "cot"), "C")
        self.assertEqual(ac.extrair_letra("Resposta: Nenhuma das alternativas anteriores.", "cot", "E"), "E")
        self.assertIsNone(ac.extrair_letra("Resposta: Nenhuma das alternativas anteriores.", "cot", None))

    def test_apurar_separa_concordancia(self):
        itens = [_item("a", "A", True, "ok"), _item("b", "B", True), _item("c", "C", False), _item("d", "D", False)]
        r = ac.apurar_registros(itens, {"a": "A", "b": "E", "c": "C", "d": None})
        self.assertEqual(r["mantidas"], 2)
        self.assertEqual(r["nucleo_entre_mantidas"], "1/2")
        self.assertEqual(r["validas_perdidas_pct"], 50.0)
        self.assertEqual(r["invalidas_descartadas_pct"], 50.0)
        self.assertEqual(r["concorda_e_ok"], "1/1")

    def test_sem_resposta_nunca_e_aprovacao(self):
        r = ac.apurar_registros([_item("x", "A", False)], {"x": None})
        self.assertEqual(r["mantidas"], 0)

    def test_critica_le_as_tres_linhas(self):
        t = "A) incorreta\nB) correta\n**CORRETAS:** B\nCONTRADIÇÃO: NÃO\nFALTA_DADO: NÃO"
        self.assertEqual(ac.extrair_critica(t), {"corretas": ["B"], "contradicao": False, "suficiente": True})
        self.assertEqual(ac.extrair_critica("CORRETAS: A, C\nCONTRADICAO: NAO\nSUFICIENTE: SIM")["corretas"], ["A", "C"])
        self.assertEqual(ac.extrair_critica("CORRETAS: NENHUMA\nCONTRADICAO: NAO\nSUFICIENTE: SIM")["corretas"], [])
        self.assertIsNone(ac.extrair_critica("CORRETAS: B\nCONTRADICAO: NAO"))  # falta linha
        self.assertIsNone(ac.extrair_critica("CORRETAS: B\nCONTRADICAO: talvez\nSUFICIENTE: SIM"))
        # v2: FALTA_DADO (SIM = problema) tem prioridade sobre SUFICIENTE da v1
        self.assertFalse(ac.extrair_critica("CORRETAS: B\nCONTRADICAO: NAO\nFALTA_DADO: SIM")["suficiente"])
        self.assertIsNone(ac.extrair_critica("CORRETAS: B\nCONTRADICAO: NAO\nFALTA_DADO: talvez"))

    def test_critica_so_aprova_uma_correta_sem_contradicao(self):
        it = _item("u3", "D", True)
        ordem = ac.ordem_alternativas("u3", True, it["alternativas"])
        exib = "ABCDE"[ordem.index("D")]
        ok = f"CORRETAS: {exib}\nCONTRADICAO: NAO\nSUFICIENTE: SIM"
        self.assertEqual(ac.decidir(ok, "critica", it, True), ("D", "uma_correta"))
        casos = {f"CORRETAS: A, B\nCONTRADICAO: NAO\nSUFICIENTE: SIM": "varias_corretas",
                 f"CORRETAS: {exib}\nCONTRADICAO: SIM\nSUFICIENTE: SIM": "contradicao",
                 f"CORRETAS: {exib}\nCONTRADICAO: NAO\nSUFICIENTE: NAO": "insuficiente",
                 "CORRETAS: NENHUMA\nCONTRADICAO: NAO\nSUFICIENTE: SIM": "nenhuma_correta",
                 "": "formato"}
        for texto, motivo in casos.items():
            self.assertEqual(ac.decidir(texto, "critica", it, True), (None, motivo))

    def test_combinar_exige_todas(self):
        itens = [_item("a", "A", True, "ok"), _item("b", "B", False), _item("c", "C", True)]
        r1, r2 = {"a": "A", "b": "B", "c": "C"}, {"a": "A", "b": "B", "c": None}
        self.assertEqual(ac.combinar(itens, r1, r2), "1/2")
        self.assertEqual(ac.combinar(itens, r1, r2, so_ok=True), "1/1")

    def test_prompt_critica_congelado(self):
        """Validação em folha nova (2026-10-25): o prompt da crítica v2 não pode mudar depois de ver o resultado.
        Se mudar de propósito, é OUTRA versão (VERSAO_CRITICA) e a validação precisa ser refeita."""
        import hashlib
        h = hashlib.sha256((ac.SISTEMA_CRITICA + "\n" + ac.FORMATO["critica"] + f"\n{ac.MAX_TOKENS['critica']}").encode()).hexdigest()
        self.assertEqual(ac.VERSAO_CRITICA, "v2")
        self.assertEqual(h[:16], "831469ec8262eeed")


if __name__ == "__main__":
    unittest.main()
