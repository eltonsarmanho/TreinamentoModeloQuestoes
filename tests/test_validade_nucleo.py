"""Testa que o critério NÚCLEO (correcao_matematica + resposta_unica + dados_suficientes)
é calculado pela apuração e que dificuldade_adequada NÃO o afeta (decisão 2026-10-14).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import amostra_humana as ah  # noqa: E402


class TestValidadeNucleo(unittest.TestCase):
    def _chave(self, ids):
        return {"itens": {qid: {"unidade": "Números", "origem": "candidato"} for qid in ids}}

    def test_nucleo_ignora_dificuldade(self):
        """Item com dificuldade=n mas núcleo todo s → valida_nucleo conta como válida."""
        validas = [{"id": "Q1", "correcao_matematica": "s", "resposta_unica": "s",
                     "dados_suficientes": "s", "aderencia_ano_habilidade": "s",
                     "contexto_plausivel": "s", "dificuldade_adequada": "n",
                     "valida_geral": "n", "comentario": ""}]
        chave = self._chave(["Q1"])
        res = ah.apurar_linhas(validas, chave)
        self.assertEqual(res["geral"]["validade_nucleo"]["s"], 1)
        self.assertEqual(res["geral"]["validade"]["s"], 0)  # valida_geral=n

    def test_nucleo_reprova_correcao(self):
        """Item com correcao_matematica=n → núcleo reprova."""
        validas = [{"id": "Q2", "correcao_matematica": "n", "resposta_unica": "s",
                     "dados_suficientes": "s", "aderencia_ano_habilidade": "s",
                     "contexto_plausivel": "s", "dificuldade_adequada": "s",
                     "valida_geral": "n", "comentario": ""}]
        chave = self._chave(["Q2"])
        res = ah.apurar_linhas(validas, chave)
        self.assertEqual(res["geral"]["validade_nucleo"]["s"], 0)

    def test_nucleo_por_origem(self):
        """validade_nucleo aparece em cada origem."""
        validas = [{"id": "Q3", "correcao_matematica": "s", "resposta_unica": "s",
                     "dados_suficientes": "s", "aderencia_ano_habilidade": "s",
                     "contexto_plausivel": "s", "dificuldade_adequada": "s",
                     "valida_geral": "s", "comentario": ""}]
        chave = self._chave(["Q3"])
        res = ah.apurar_linhas(validas, chave)
        self.assertIn("validade_nucleo", res["por_origem"]["candidato"])

    def test_constantes_nucleo(self):
        self.assertEqual(ah.NUCLEO, ("correcao_matematica", "resposta_unica", "dados_suficientes"))


if __name__ == "__main__":
    unittest.main()
