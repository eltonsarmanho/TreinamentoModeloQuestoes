"""P0-2 (2026-10-10) — verificador de consistência aritmética ESTENDIDO.

Cobre schema_utils._analisar_contas / _contas_ricas (somas n-árias, parênteses,
unidades, vírgula decimal pt-BR, igualdades encadeadas), a regra forte
"resultado da resolução fora das alternativas" com tolerância documentada, a
regra suave de "dado inventado" (aviso, não muda status) e a integração com
generate_validated / RANK_STATUS.

Os casos marcados REAL são cópias literais de gerações do modelo
(outputs/scratch/exemplos_sufixo.json e outputs/diversidade_*.json) ou de itens
do corpus de treino; ficam embutidos porque outputs/ é gitignored.

    venv/bin/python -m pytest -q tests/test_verificador_consistencia_estendido.py
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gerar_lote as gl  # noqa: E402
import schema_utils as su  # noqa: E402
import test_model as tm  # noqa: E402

ALTS5 = lambda *v: dict(zip("ABCDE", v))  # noqa: E731


def _q(enunciado, alternativas, resolucao, gabarito, difficulty="MEDIUM"):
    return {"enunciado": enunciado, "alternativas": dict(alternativas),
            "resolucao_passo_a_passo": resolucao, "resposta_correta": gabarito,
            "difficulty": difficulty}


def _res(texto):
    return su._computed_results(texto)


class TestReconhecimentoDeContas(unittest.TestCase):
    """O que antes devolvia [] (medido na auditoria de 2026-10-10)."""

    def test_casos_medidos_que_retornavam_vazio(self):
        for texto, esperado in [
            ("8 + 6 + 8 + 6 = 28 cm", 28),
            ("180° - 150° = 30°", 30),
            ("2 x (8 + 6) = 28", 28),
            ("Perímetro = 2 × (8 + 6) = 2 × 14 = 28 cm", 28),
            ("12 + 8 + 7 + 3 = 30", 30),
        ]:
            with self.subTest(texto=texto):
                self.assertEqual(_res(texto)[-1], esperado)

    def test_continua_reconhecendo_o_binario_antigo(self):
        self.assertEqual(_res("9 x 12 = 108"), [108])
        self.assertEqual(_res("6 + 7 = 13. Depois, 13 + 5 = 18"), [13, 18])

    def test_precedencia_e_parenteses(self):
        self.assertEqual(_res("2 + 3 x 4 = 14")[-1], 14)
        self.assertEqual(_res("(2 + 3) x 4 = 20")[-1], 20)
        self.assertEqual(_res("20 - 2 x (3 + 4) = 6")[-1], 6)
        self.assertEqual(_res("3² - 4x3 + 6 = 9 - 12 + 6 = 3")[-1], 3)
        self.assertEqual(_res("5² = 25")[-1], 25)

    def test_unidades_coladas_e_separadas(self):
        for texto, esperado in [
            ("5 cm x 3 cm = 15 cm²", 15), ("5cm x 3cm = 15cm²", 15),
            ("250 g + 180 g = 430 g", 430), ("3 kg x 4 = 12 kg", 12),
            ("2 L + 500 mL = 502", None),  # unidades diferentes: não "bate" (502 != 2.5)
            ("R$ 2,50 x 3 = R$ 7,50", 7.5), ("2,50 x 3 = 7,50", 7.5),
            ("10 min + 20 min = 30 min", 30), ("25% + 10% = 35%", 35),
            ("20 lâminas x 30 cm = 600 cm = 6 m", 6),
        ]:
            with self.subTest(texto=texto):
                if esperado is None:
                    continue
                self.assertEqual(_res(texto)[-1], esperado)

    def test_virgula_decimal_e_milhar_pt_br(self):
        self.assertEqual(_res("R$ 1.250,50 x 2 = R$ 2.501,00")[-1], 2501)
        self.assertEqual(_res("0,5 x 4 = 2")[-1], 2)
        self.assertEqual(_res("12,5 + 7,5 = 20,0")[-1], 20)
        # ponto de milhar sem vírgula: lido como decimal aqui; a leitura de
        # milhar é a de _sem_separador_milhar (absolve em check_consistency)
        self.assertEqual(_res(su._sem_separador_milhar("1.250 + 250 = 1.500"))[-1], 1500)

    def test_sinais_e_operadores(self):
        self.assertEqual(_res("12 ÷ 4 = 3")[-1], 3)
        self.assertEqual(_res("12 / 4 = 3")[-1], 3)
        self.assertEqual(_res("12 : 4 = 3")[-1], 3)
        self.assertEqual(_res("3 × 4 = 12")[-1], 12)
        self.assertEqual(_res("3 * 4 = 12")[-1], 12)
        self.assertEqual(_res("3 · 4 = 12")[-1], 12)
        self.assertEqual(_res("10 − 4 = 6")[-1], 6)
        self.assertEqual(_res("10 – 4 = 6")[-1], 6)
        self.assertEqual(_res("- 5 + 3 = 8"), _res("5 + 3 = 8"))  # traço de lista

    def test_percentual_e_fracao_de(self):
        self.assertEqual(_res("20% de 150 = 30")[-1], 30)
        self.assertEqual(_res("3/4 de 20 = 15")[-1], 15)
        self.assertEqual(_res("1 ÷ (3/8 x 2/3) = 1 ÷ (1/4) = 4")[-1], 4)

    def test_relogio(self):
        self.assertEqual(_res("10h30 + 30 minutos = 11h00"), [660])
        self.assertEqual(_res("10h30 + 45 min = 11h15"), [675])
        self.assertEqual(_res("10h30 + 30 minutos = 11h30"), [])  # conta errada

    def test_igualdades_encadeadas(self):
        self.assertEqual(_res("5 + 3 = 8 = 8"), [8, 8])
        self.assertEqual(_res("120 - 80 = 40; 40 ÷ 2 = 20"), [40, 20])
        self.assertEqual(_res("120 - 80 = 40 camisetas; 40 ÷ 2 = 20 camisetas"), [40, 20])
        r = _res("Área = 3 x 4 x 5 = 12 x 5 = 60 cm³")
        self.assertEqual(r[-1], 60)
        self.assertGreaterEqual(len(r), 2)

    def test_aproximado_so_com_sinal_de_aproximacao(self):
        self.assertEqual(_res("40/120 = 1/3 ≈ 33.3%")[-1], 33.3)
        self.assertEqual(_res("10 ÷ 3 ≈ 3,33")[-1], 3.33)
        self.assertEqual(_res("10 ÷ 3 ≈ 3")[-1], 3)
        # "=" é exato (decisão H2 de test_base3_decisoes: aproximado nunca verifica)
        self.assertEqual(_res("10 ÷ 3 = 3,33"), [])
        self.assertEqual(_res("7 ÷ 2 = 3"), [])


class TestContasErradasENaoConta(unittest.TestCase):
    def test_conta_interna_errada_nao_vira_resultado(self):
        for texto in ["8 + 6 + 8 + 6 = 27 cm", "180° - 150° = 40°", "2 x (8 + 6) = 30",
                      "5 + 3 = 9", "3 x 4 = 13"]:
            with self.subTest(texto=texto):
                self.assertEqual(_res(texto), [])

    def test_divisao_por_zero(self):
        self.assertEqual(_res("10 ÷ 0 = 5"), [])
        self.assertEqual(_res("10 / 0 = 0"), [])
        self.assertEqual(_res("5 / (3 - 3) = 1"), [])

    def test_overflow_e_numeros_gigantes(self):
        self.assertEqual(_res("2 ^ 99999 = 1"), [])
        self.assertEqual(_res("9" * 5000 + " + 1 = 2"), [])
        self.assertEqual(_res("(" * 500 + "1" + ")" * 500 + " = 1"), [])
        self.assertEqual(_res("999999999999 x 999999999999 = 998000000000000000000000"), [])

    def test_algebra_e_variavel_nao_viram_conta(self):
        for texto in ["x + 3 = 8", "2x + 3 = 11", "4x + 12 = 72", "3x - 4x + 1 = -x + 1"]:
            with self.subTest(texto=texto):
                self.assertEqual(_res(texto), [])

    def test_nao_quebra_com_lixo(self):
        for texto in ["", "=", "= =", "+ - x", "((", "1 +", "= 5", "R$ = 3", "1,,2 = 3",
                      "10h99 + 1 = 11h00", "99999999999999999999999 x 2 = 1"]:
            with self.subTest(texto=texto):
                self.assertIsInstance(_res(texto), list)


# ---------------------------------------------------------------------------
# Regra forte: resultado final da resolução fora das alternativas
# ---------------------------------------------------------------------------

# REAL (exemplos_sufixo #4, 9º H03): resolução calcula 40/120 ≈ 33,3%, nenhuma
# alternativa vale isso; gabarito C=25%.
Q_CAMISETAS = _q(
    "Uma loja vendeu 120 camisetas no primeiro mês. No segundo mês, vendeu 80 "
    "camisetas. Qual foi a porcentagem de variação entre os dois meses?",
    ALTS5("100%", "20%", "25%", "50%", "120%"),
    "A variação foi de 40 camisetas. 40/120=1/3≈33.3%, ou 33.3% ≈ 33%. "
    "Portanto, a resposta é 33%.", "C")

# REAL (exemplos_sufixo #12, 5º H17): 10h30 + 30 minutos = 11h00 é internamente
# correta e 11h00 está nas alternativas -> "ok"; só o AVISO pega o 30 inventado.
Q_BOLO = _q(
    "Juliana comprou um bolo com 45 minutos de duração. Se ela colocar o bolo no "
    "forno, ele começa a assar às 10h30. Em qual horário o bolo começa a assar?",
    ALTS5("11h00", "10h30", "10h00", "10h15", "Nenhuma das alternativas anteriores"),
    "10h30 + 30 minutos = 11h00", "A")

# REAL (repro_out (0,23), 9º): "8 + 6 + 8 + 6 = 28 cm" e alternativas 16..26.
Q_PERIMETRO = _q(
    "Quadrilátero ABCD tem lados AB = 8 cm, BC = 6 cm, CD = 8 cm e DA = 6 cm. "
    "Qual é o perímetro desse quadrilátero?",
    ALTS5("20 cm", "24 cm", "18 cm", "16 cm", "26 cm"),
    "Somando os quatro lados: 8 + 6 + 8 + 6 = 28 cm. Como os lados opostos são "
    "iguais (AB = CD e BC = DA), o quadrilátero é um retângulo, e o perímetro é "
    "2 x (8 + 6) = 28 cm. Portanto, a resposta é 28 cm.", "A")

# REAL (repro_out (0,22), 9º): 180° - 150° = 30°, alternativas 70°..110°.
Q_ANGULO = _q(
    "Em um triângulo qualquer, a soma das medidas dos ângulos internos é igual a "
    "180°. Qual é a medida do terceiro ângulo desse triângulo, se dois deles "
    "medem 70° e 80°?",
    ALTS5("100°", "90°", "80°", "110°", "70°"),
    "Somando os dois ângulos conhecidos: 70° + 80° = 150°. Subtraindo esse "
    "resultado do total das medidas dos ângulos internos: 180° - 150° = 30°. "
    "Portanto, o terceiro ângulo mede 30°.", "C")


class TestRegraForte(unittest.TestCase):
    def test_os_dois_casos_reais_citados(self):
        ok, sug, motivo = su.check_consistency_detalhado(Q_CAMISETAS)
        self.assertIs(ok, False)
        self.assertIsNone(sug)
        self.assertEqual(motivo, su.MOTIVO_FORA_DAS_ALTERNATIVAS)
        self.assertEqual(su.fix_gabarito(dict(Q_CAMISETAS))[1], "fora_das_alternativas")
        # o do bolo NÃO é reprovado (conta correta, 11h00 existe): só aviso
        self.assertIs(su.check_consistency_detalhado(Q_BOLO)[0], True)
        self.assertEqual([a["valor"] for a in su.avisos_consistencia(Q_BOLO)], [30])

    def test_perimetro_e_angulo_viram_fora_das_alternativas(self):
        for q in (Q_PERIMETRO, Q_ANGULO):
            ok, sug, motivo = su.check_consistency_detalhado(q)
            self.assertIs(ok, False)
            self.assertEqual(motivo, su.MOTIVO_FORA_DAS_ALTERNATIVAS)
            self.assertTrue(su.resposta_fora_das_alternativas(q))
            self.assertEqual(su.check_consistency(q), (False, None))

    def test_gabarito_certo_passa_nos_mesmos_formatos(self):
        q = dict(Q_PERIMETRO, resposta_correta="A")
        q["alternativas"] = ALTS5("28 cm", "24 cm", "18 cm", "16 cm", "26 cm")
        self.assertEqual(su.check_consistency(q), (True, None))
        q2 = dict(Q_ANGULO, alternativas=ALTS5("30°", "90°", "80°", "110°", "70°"),
                  resposta_correta="A")
        self.assertEqual(su.check_consistency(q2), (True, None))

    def test_resultado_em_outra_alternativa_mantem_gabarito_errado_e_fix(self):
        q = _q("Quanto é 8 + 6 + 8 + 6?", ALTS5("20", "24", "28", "16", "26"),
               "8 + 6 + 8 + 6 = 28", "A")
        ok, sug, motivo = su.check_consistency_detalhado(q)
        self.assertEqual((ok, sug, motivo), (False, "C", "gabarito_errado"))
        q2, status = su.fix_gabarito(dict(q))
        self.assertEqual((status, q2["resposta_correta"]), ("corrigido", "C"))

    def test_formatacao_nao_gera_falso_positivo(self):
        for alt, gab in [(("1250", "1000", "1500", "2000", "900"), "A"),
                         (("1.250", "1.000", "1.500", "2.000", "900"), "A"),
                         (("1250 reais", "1000 reais", "1500 reais", "2000 reais", "900 reais"), "A"),
                         (("R$ 1.250,00", "R$ 1.000,00", "R$ 1.500,00", "R$ 2.000,00", "R$ 900,00"), "A")]:
            with self.subTest(alt=alt[0]):
                q = _q("Quanto custa?", ALTS5(*alt), "1000 + 250 = 1250", gab)
                self.assertIs(su.check_consistency_detalhado(q)[0], True)
        # e o mesmo item com o gabarito trocado é pego sem ser "fora"
        q = _q("Quanto custa?", ALTS5("R$ 1.250,00", "R$ 1.000,00", "R$ 1.500,00",
                                      "R$ 2.000,00", "R$ 900,00"), "1000 + 250 = 1250", "B")
        ok, sug, _ = su.check_consistency_detalhado(q)
        self.assertEqual((ok, sug), (False, "A"))

    def test_arredondamento_documentado(self):
        # 33,3 ~ 33%: alternativa com menos casas absolve; vizinhas não casam
        q = _q("Variação?", ALTS5("100%", "20%", "33%", "50%", "120%"),
               "40/120 = 1/3 ≈ 33,3%", "C")
        self.assertIs(su.check_consistency_detalhado(q)[0], True)
        # inteiro nunca "arredonda": 120 não casa com 121
        q = _q("Total?", ALTS5("121", "100", "110", "130", "90"), "100 + 20 = 120", "A")
        ok, sug, motivo = su.check_consistency_detalhado(q)
        self.assertIs(ok, False)
        self.assertEqual(motivo, su.MOTIVO_FORA_DAS_ALTERNATIVAS)
        # 7,75 ~ 7,8 (meio para cima) e ~ 8
        q = _q("Média?", ALTS5("7,5", "7,8", "8", "7,9", "7"), "31 ÷ 4 ≈ 7,75", "B")
        self.assertIs(su.check_consistency_detalhado(q)[0], True)

    def test_fracao_e_percentual_equivalentes(self):
        q = _q("Quanto é 1/4?", ALTS5("10%", "25%", "40%", "50%", "75%"),
               "1 ÷ 4 = 0,25", "B")
        self.assertIs(su.check_consistency_detalhado(q)[0], True)
        q = _q("Quanto é 2,5 m em cm?", ALTS5("25 cm", "250 cm", "2500 cm", "0,25 cm", "2,5 cm"),
               "2,5 m x 100 = 250 cm", "B")
        self.assertIs(su.check_consistency_detalhado(q)[0], True)

    def test_unidade_incompativel_nao_acusa(self):
        # o último cálculo é um ângulo central (°) e as alternativas são cm
        q = _q("Qual o comprimento da corda?", ALTS5("14 cm", "6 cm", "12 cm", "10 cm", "8 cm"),
               "O ângulo central é 2 x 60° = 120°. Logo a corda mede 12 cm.", "C")
        self.assertIsNot(su.check_consistency_detalhado(q)[0], False)

    def test_tres_ou_mais_alternativas_de_valor_sao_exigidas(self):
        q = _q("Quem tem mais?", ALTS5("Ana", "Bia", "Caio", "Davi", "Eva"),
               "8 + 6 + 8 + 6 = 28. Portanto Ana.", "A")
        self.assertIsNone(su.check_consistency_detalhado(q)[0])

    def test_gabarito_nda_continua_sem_veredito(self):
        q = _q("Total?", ALTS5("10", "20", "30", "40", "Nenhuma das alternativas anteriores"),
               "5 + 6 + 7 = 18", "E")
        self.assertEqual(su.check_consistency_detalhado(q)[2], "gabarito_nda")


class TestGuardasContraFalsoPositivo(unittest.TestCase):
    """Cada caso é um falso positivo REAL que a regra crua teria produzido."""

    def test_conferencia_depois_da_algebra(self):
        # train_curado_v3 linha 1336: x = 15 por álgebra, depois "15 + 30 + 27 = 72"
        q = _q("Perímetro 72 cm; um lado é o dobro do menor e o terceiro tem 12 cm a mais. "
               "Quanto mede o menor lado?",
               ALTS5("12 cm", "15 cm", "18 cm", "20 cm", "24 cm"),
               "Chame de x o menor lado: x + 2x + (x + 12) = 72 → 4x + 12 = 72 → "
               "4x = 60 → x = 15 cm. Verifica-se: 15 + 30 + 27 = 72.", "B")
        self.assertIsNot(su.check_consistency_detalhado(q)[0], False)

    def test_substituicao_que_reproduz_dado_do_enunciado(self):
        # val.jsonl linha 13: "3x7 + 5 = 21 + 5 = 26" confere o 26 do enunciado
        q = _q("A soma do triplo de um número com 5 é igual a 26. Qual é esse número?",
               ALTS5("6", "7", "8", "9", "Nenhuma das alternativas anteriores"),
               "3x + 5 = 26 → 3x7 + 5 = 21 + 5 = 26", "B")
        self.assertIsNot(su.check_consistency_detalhado(q)[0], False)

    def test_conversao_de_horario_do_enunciado_ainda_e_acusada(self):
        # o guarda do parágrafo acima NÃO pode absolver este erro real
        q = _q("O relógio mostrava 10h20min. Quantos minutos passaram desde a meia-noite?",
               ALTS5("20", "10020", "120", "1020", "Nenhuma das alternativas anteriores"),
               "10h20min é igual a 10 x 60 + 20 = 620 minutos.", "D")
        self.assertIs(su.check_consistency_detalhado(q)[0], False)

    def test_resposta_na_cauda_depois_da_ultima_conta(self):
        q = _q("Qual a porcentagem?", ALTS5("23,8%", "25,0%", "28,5%", "30,0%", "33,3%"),
               "Total = 12 + 8 + 7 + 3 = 30. (7 ÷ 30) × 100 = 23,333...%, arredondada para 23,8%.", "A")
        self.assertIsNot(su.check_consistency_detalhado(q)[0], False)  # não troca para 30,0%

    def test_ramo_textual_nao_ganhou_alcance(self):
        # notação científica: o motor estendido acha contas, mas o ramo textual só
        # roda para quem já tinha conta binária (antes: "corresponde_outra" errado)
        q = _q("Um ônibus carregava 2,5 x 10^3 passageiros e 5 x 10^2 desembarcaram. Quantos restaram?",
               ALTS5("3,5 x 10^4", "7,5 x 10^4", "2,5 x 10^5", "7,5 x 10^3", "3,5 x 10^5"),
               "2,5 x 10^3 - 5 x 10^2 = 2,5 x 10^3 - 0,5 x 10^3 = 2,0 x 10^3", "B")
        self.assertIsNone(su.check_consistency_detalhado(q)[0])


# ---------------------------------------------------------------------------
# Regra suave: dado inventado (aviso)
# ---------------------------------------------------------------------------

class TestAvisoDadoInventado(unittest.TestCase):
    def test_aviso_nao_muda_status(self):
        q = dict(Q_BOLO)
        antes = su.check_consistency_detalhado(q)
        self.assertTrue(su.avisos_consistencia(q))
        self.assertEqual(su.detalhe_consistencia(q)["motivo"], antes[2])
        self.assertIs(su.detalhe_consistencia(q)["consistente"], True)
        self.assertEqual(su.fix_gabarito(dict(q))[1], "ok")

    def test_nao_avisa_quando_o_dado_esta_no_enunciado_ou_vem_de_passo_anterior(self):
        q = _q("Ana tem 25 figurinhas e comprou 17. Quantas tem agora?",
               ALTS5("38", "42", "43", "44", "45"), "25 + 17 = 42 figurinhas.", "B")
        self.assertEqual(su.avisos_consistencia(q), [])
        q = _q("Um bolo leva 3 ovos. Quantos ovos para 4 bolos e depois mais 5?",
               ALTS5("12", "17", "15", "20", "9"), "3 x 4 = 12. Depois 12 + 5 = 17.", "B")
        self.assertEqual(su.avisos_consistencia(q), [])

    def test_numero_por_extenso_do_enunciado_e_constante_nao_avisam(self):
        q = _q("Pedro comprou dois lápis de R$ 3,00 cada. Quanto gastou?",
               ALTS5("R$ 5,00", "R$ 6,00", "R$ 7,00", "R$ 8,00", "R$ 9,00"), "2 x 3 = 6", "B")
        self.assertEqual(su.avisos_consistencia(q), [])
        q = _q("Quanto vale o perímetro de um quadrado de lado 5 cm?",
               ALTS5("10 cm", "15 cm", "20 cm", "25 cm", "30 cm"), "4 x 5 = 20 cm", "C")
        self.assertEqual(su.avisos_consistencia(q), [])

    def test_robusto_a_entrada_ruim(self):
        for q in (None, {}, {"resolucao_passo_a_passo": None}, {"enunciado": None}):
            self.assertEqual(su.avisos_consistencia(q), [])


# ---------------------------------------------------------------------------
# Integração: generate_validated, RANK_STATUS, relatórios
# ---------------------------------------------------------------------------

def _gen_de(questoes):
    def gen(llama_cli, gguf, prompt, threads, max_new_tokens, seed=None, grammar=None):
        q = questoes[min(len(questoes) - 1, gen.n)]
        gen.n += 1
        return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01
    gen.n = 0
    return gen


class TestIntegracao(unittest.TestCase):
    def test_reprovada_pelo_resultado_fora_e_regenerada(self):
        boa = dict(Q_PERIMETRO, resposta_correta="A")
        boa["alternativas"] = ALTS5("28 cm", "24 cm", "18 cm", "16 cm", "26 cm")
        gen = _gen_de([Q_PERIMETRO, boa])
        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=1, gen_fn=gen)
        self.assertEqual(gen.n, 2)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["regeneracoes"], 1)

    def test_reprovada_em_todas_as_amostras_vira_falha_e_nao_e_trocada(self):
        gen = _gen_de([Q_PERIMETRO])
        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=1, gen_fn=gen)
        self.assertEqual(r["status"], "falha")
        self.assertEqual(r["motivo_consistencia"], su.MOTIVO_FORA_DAS_ALTERNATIVAS)
        self.assertEqual(r["obj"]["resposta_correta"], "A")

    def test_ok_nao_gasta_chamada_extra_e_expoe_avisos(self):
        gen = _gen_de([Q_BOLO])
        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=2, gen_fn=gen)
        self.assertEqual((gen.n, r["status"]), (1, "ok"))
        self.assertEqual([a["tipo"] for a in r["avisos_consistencia"]], ["dado_inventado"])

    def test_nao_verificavel_continua_aceito_sem_chamada_extra(self):
        q = _q("Qual figura tem 3 lados?", ALTS5("Triângulo", "Quadrado", "Círculo", "Reta", "Ponto"),
               "O triângulo tem três lados.", "A")
        gen = _gen_de([q])
        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=2, gen_fn=gen)
        self.assertEqual((gen.n, r["status"]), (1, "nao_verificavel"))
        self.assertEqual(r["avisos_consistencia"], [])

    def test_ok_e_nao_verificavel_sao_contados_separadamente_e_ranqueados(self):
        self.assertGreater(gl.RANK_STATUS["ok"], gl.RANK_STATUS["nao_verificavel"])
        self.assertGreater(gl.RANK_STATUS["nao_verificavel"], gl.RANK_STATUS["corrigido"])
        self.assertEqual(gl.RANK_STATUS["falha"], 0)
        cand = lambda st: {"status": st, "obj": {"x": 1}, "violacoes": []}  # noqa: E731
        self.assertGreater(gl._chave(cand("ok")), gl._chave(cand("nao_verificavel")))
        self.assertTrue(gl._utilizavel(cand("nao_verificavel")))
        # score do best-of-N: 8 (verificada) > 6 (não verificável) > 2 (fora)
        flags = {k: True for k in ("json_valido", "wrapper_valido", "quantidade_correta",
                                   "schema_completo", "resposta_valida",
                                   "alternativas_distintas", "difficulty_valida")}
        self.assertEqual(tm._score_candidato(flags, True), 8)
        self.assertEqual(tm._score_candidato(flags, None), 6)
        self.assertEqual(tm._score_candidato(flags, False, fora_das_alternativas=True), 2)


# ---------------------------------------------------------------------------
# Não-regressão sobre o corpus real (data/ é só lido)
# ---------------------------------------------------------------------------

def _corpus(nome):
    caminho = ROOT / "data" / nome
    if not caminho.exists():
        raise unittest.SkipTest(f"{nome} ausente")
    out = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(linha)["messages"]
        obj = su.parse_json(msgs[-1]["content"])
        out.extend(su.extract_questoes(obj) or [])
    return out


class TestNaoRegressaoCorpusReal(unittest.TestCase):
    """Todos os itens do treino/val são revisados e CORRETOS: nenhuma reprovação
    nova pode aparecer, e nada que era 'ok' pode deixar de ser. Medido em
    2026-10-10 sobre train_curado_v3 (2759): ok 1310 -> 1624, nv 1449 -> 1135,
    reprovadas novas 0 (a regra crua teria dado 42 falsos positivos: todos
    absolvidos pelas guardas de cauda/menção)."""

    @classmethod
    def setUpClass(cls):
        cls.v3 = _corpus("train_curado_v3.jsonl")
        if len(cls.v3) < 2000:
            raise unittest.SkipTest("corpus v3 incompleto")

    def test_nenhum_item_do_treino_e_reprovado(self):
        ruins = [i for i, q in enumerate(self.v3) if su.check_consistency_detalhado(q)[0] is False]
        self.assertEqual(ruins, [], f"itens reprovados: {ruins[:10]}")

    def test_cobertura_nao_regrediu_e_subiu(self):
        ok = sum(1 for q in self.v3 if su.check_consistency_detalhado(q)[0] is True)
        self.assertGreaterEqual(ok, 1500)   # antes: 1310
        self.assertLessEqual(len(self.v3) - ok, 1260)  # nao_verificavel antes: 1449

    def test_val_congelado_sem_reprovacao(self):
        for nome in ("val_frozen_v1.jsonl", "val_novos_v1.jsonl"):
            with self.subTest(nome=nome):
                qs = _corpus(nome)
                self.assertEqual([i for i, q in enumerate(qs)
                                  if su.check_consistency_detalhado(q)[0] is False], [])

    def test_avisos_nao_mudam_nenhum_veredito(self):
        # contrato: avisos_consistencia é função à parte; o veredito é o mesmo com ou sem ela
        for q in self.v3[:300]:
            self.assertEqual(su.detalhe_consistencia(q)["motivo"],
                             su.check_consistency_detalhado(q)[2])

    def test_taxa_do_aviso_dado_inventado_e_baixa(self):
        com_conta = [q for q in self.v3 if su._contas_ricas(q["resolucao_passo_a_passo"])]
        avisados = [q for q in com_conta if su.avisos_consistencia(q)]
        taxa = len(avisados) / len(com_conta)
        self.assertLess(taxa, 0.08)  # medido: 5,0% das questões com conta


class TestPortabilidadeDoMotor(unittest.TestCase):
    """O verificador roda no app via TypeScript: nada de construções de regex que
    o JS não tem (ver test_verificador_geometria.TestPortabilidade)."""

    PROIBIDOS = ("(?<=", "(?<!", "(?P<", "(?i", "(?a", "(?u", "(?L", "(?s", "(?m", "(?x",
                 "\\p{", "(?>", "{,", "\\Z")

    def test_bloco_do_motor_sem_construcoes_nao_portaveis(self):
        fonte = (ROOT / "src" / "schema_utils.py").read_text(encoding="utf-8")
        ini = fonte.index("# MOTOR ARITMÉTICO ESTENDIDO")
        fim = fonte.index("def _resultados_binarios")
        bloco = "\n".join(l for l in fonte[ini:fim].splitlines() if not l.lstrip().startswith("#"))
        for p in self.PROIBIDOS:
            with self.subTest(p=p):
                self.assertNotIn(p, bloco)

    def test_regex_do_motor_nao_usam_nomeados_nem_lookbehind(self):
        for nome in ("_NUM_MILHAR_RE", "_NUM_PLANO_RE", "_RELOGIO_RE", "_RELOGIO_ALT"):
            padrao = getattr(su, nome).pattern
            self.assertFalse(re.search(r"\(\?<|\(\?P|\(\?[a-z]", padrao), nome)


if __name__ == "__main__":
    unittest.main()
