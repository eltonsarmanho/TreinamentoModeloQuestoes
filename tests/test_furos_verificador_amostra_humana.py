"""Furos do verificador aritmético achados na amostra humana cega (2026-10-14).

Dois itens do v7b seed 44 que o professor reprovou em correção matemática e que o
verificador aprovou como `valor_bate`. Causa: a regra (1) de `check_consistency_detalhado`
("o valor do gabarito é resultado de ALGUMA conta da resolução, inclusive passos
intermediários", preservada byte a byte para não regredir questões hoje `ok`) absolve
um resultado parcial mesmo quando o resultado FINAL cai fora de todas as alternativas.

Tratamento (Fase 3.0): a regra (1) NÃO foi alterada, porque uma regra mais estrita atingiria ~2,2%
dos itens de train_curado_v7b, em sua maioria legítimos. Em vez disso, `avisos_gabarito` emite o
aviso não bloqueante "gabarito_so_em_passo_intermediario". Tornar o aviso bloqueante exige
veredito reprovado e revisão da amostra (ver ANALISE_ERROS_CORRECAO.md).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import schema_utils as su  # noqa: E402

ONIBUS = {  # amostra humana Q6415: soma 1250, gabarito C = 1000 (parcial 500+500)
    "enunciado": ("Uma empresa de ônibus registrou os seguintes valores de passageiros durante uma "
                  "viagem: 12 horas - 200 passageiros, 2 horas - 300 passageiros, 1 hora - 500 "
                  "passageiros, 3 horas - 250 passageiros. Quantos passageiros havia ao longo do "
                  "trajeto, considerando apenas os dados da tabela?"),
    "alternativas": {"A": "100 passageiros.", "B": "300 passageiros.", "C": "1000 passageiros.",
                     "D": "800 passageiros.", "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo": (
        "A pergunta pede o total de passageiros ao longo do trajeto, usando apenas os dados da "
        "tabela. Para resolver, somamos os passageiros em cada intervalo: 200 passageiros + 300 "
        "passageiros + 500 passageiros + 250 passageiros. Calculando: 200 + 300 = 500; 500 + 500 = "
        "1000; 1000 + 250 = 1250. Portanto, havia 1250 passageiros ao longo do trajeto."),
    "resposta_correta": "C", "difficulty": "EASY",
}

PAO = {  # amostra humana Q1944: resolução dá 45π - 1500; D = 1500 - 300π não bate
    "enunciado": ("Em uma padaria, um pão de forma tem formato de bloco retangular com 20 cm de "
                  "comprimento, 15 cm de largura e 5 cm de altura. Para embalar, o dono usa papelão "
                  "em formato de cilindro com raio 3 cm e altura igual à altura do pão. Qual é a "
                  "diferença, em cm³, entre o volume do cilindro e o volume do pão de forma?"),
    "alternativas": {"A": "250π - 1500", "B": "300π - 1500", "C": "1500 - 250π",
                     "D": "1500 - 300π", "E": "300 - 250π"},
    "resolucao_passo_a_passo": (
        "Volume do pão: 20 × 15 × 5 = 1500 cm³. Volume do cilindro: π × 3² × 5 = 45π cm³. "
        "Diferença: 45π - 1500. Assim, a resposta é 45π - 1500, ou seja, 1500 - 45π (alternativa D)."),
    "resposta_correta": "D", "difficulty": "MEDIUM",
}


class TestAvisoGabaritoIntermediario(unittest.TestCase):
    def test_veredito_nao_muda(self):
        """O aviso é só sinal: o verificador continua aprovando (regra (1) preservada)."""
        self.assertEqual(su.check_consistency_detalhado(ONIBUS), (True, None, "valor_bate"))
        self.assertEqual(su.check_consistency_detalhado(PAO), (True, None, "valor_bate"))

    def test_soma_parcial_dispara_o_aviso(self):
        av = su.avisos_gabarito(ONIBUS)
        self.assertEqual([a["tipo"] for a in av], ["gabarito_so_em_passo_intermediario"])
        self.assertEqual(av[0]["final"], 1250)

    def test_pi_simbolico_dispara_o_aviso(self):
        self.assertEqual([a["tipo"] for a in su.avisos_gabarito(PAO)],
                         ["gabarito_so_em_passo_intermediario"])

    def test_gabarito_igual_ao_final_nao_avisa(self):
        q = dict(ONIBUS, resposta_correta="C",
                 alternativas=dict(ONIBUS["alternativas"], C="1250 passageiros."))
        self.assertEqual(su.avisos_gabarito(q), [])

    def test_entrada_invalida_nao_quebra(self):
        self.assertEqual(su.avisos_gabarito(None), [])
        self.assertEqual(su.avisos_gabarito({}), [])

    def test_aparece_no_detalhe_sem_alterar_veredito(self):
        d = su.detalhe_consistencia(ONIBUS)
        self.assertTrue(d["consistente"])
        self.assertIn("gabarito_so_em_passo_intermediario", [a["tipo"] for a in d["avisos"]])

    def test_avisos_consistencia_antigo_nao_foi_alterado(self):
        """Quem conta `dado_inventado` (triar_resolucoes) não deve ver o aviso novo."""
        self.assertNotIn("gabarito_so_em_passo_intermediario",
                         [a["tipo"] for a in su.avisos_consistencia(ONIBUS)])


if __name__ == "__main__":
    unittest.main()
