"""Regressão da Fase 0.1/0.2: perguntas de COMPARAÇÃO e RESPOSTA FORA DAS ALTERNATIVAS.

Todas as questões deste arquivo são REAIS, copiadas literalmente de
outputs/diversidade_*.json e de data/train_curado.jsonl (a origem está no
docstring de cada teste). Nenhum enunciado foi inventado: o valor do arquivo
está justamente em travar o comportamento nos casos que a medição de 2026-09
mostrou estarem errados.

Contexto medido antes da mudança (916 questões únicas dos 14 relatórios):
das 16 questões marcadas "inconsistente", 4 eram FALSO POSITIVO — 25% de
imprecisão. Todas pelo mesmo motivo: a alternativa é um RÓTULO ("Área C",
"A caixa quadrada", "A"), não o VALOR da resposta, e _leading_number lia um
número irrelevante. Do outro lado, questões cujo resultado final não está em
alternativa nenhuma (padaria: 150-80=70 com alternativas 60/100/110/120/130)
eram devolvidas como "inconsistente" sem sugestão, indistinguíveis de um
gabarito meramente trocado — embora sejam IRRECUPERÁVEIS por troca de letra.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from schema_utils import (  # noqa: E402
    MOTIVO_FORA_DAS_ALTERNATIVAS,
    check_consistency,
    check_consistency_detalhado,
    classe_alternativa,
    fix_gabarito,
    resposta_fora_das_alternativas,
)


def _q(enunciado, alternativas, resolucao, gabarito, difficulty="MEDIUM"):
    return {
        "enunciado": enunciado,
        "alternativas": dict(alternativas),
        "resolucao_passo_a_passo": resolucao,
        "resposta_correta": gabarito,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Os 4 FALSOS POSITIVOS conhecidos (Problema A) — todos devem virar ok=True.
# ---------------------------------------------------------------------------

JARDINS = _q(
    "Em uma caminhada ao parque, João e Ana decidiram medir as áreas de dois "
    "jardins. Um jardim tem 12 metros de comprimento e 10 metros de largura. O "
    "outro jardim tem 14 metros de comprimento e 8 metros de largura. Qual "
    "jardim tem maior área?",
    {"A": "Jardim com 12 x 10 m².", "B": "Jardim com 14 x 8 m².",
     "C": "Jardim com 12 x 8 m².", "D": "Jardim com 10 x 8 m².",
     "E": "Nenhuma das alternativas anteriores"},
    "Área do primeiro jardim: 12 x 10 = 120 m². Área do segundo jardim: "
    "14 x 8 = 112 m². O jardim com 12 x 10 m² tem maior área.",
    "A", "HARD",
)

TIMES = _q(
    "No campeonato de futebol, os times A e B usaram malhas quadriculadas para "
    "desenhar o campo. O time A tem 3 quadricula em largura e 6 em comprimento. "
    "O time B tem 5 quadricula em largura e 2 em comprimento. Qual das duas "
    "figuras tem maior área?",
    {"A": "A", "B": "B", "C": "Ambas têm a mesma área.",
     "D": "Nenhuma das alternativas anteriores",
     "E": "Nenhuma das alternativas anteriores"},
    "Área do time A: 3x6=18. Área do time B: 5x2=10. A área do time A é maior.",
    "A", "HARD",
)

AREAS_ABC = _q(
    "Na preparação do campo de futebol, o técnico decidiu separar 3 regiões: uma "
    "área A, uma área B e uma área C. A área A tem 10 quadradinhos, área B tem "
    "24 quadradinhos e área C tem 36 quadradinhos. Em qual dessas áreas a "
    "superfície é a mais extensa?",
    {"A": "Área A", "B": "Área B", "C": "Área C",
     "D": "Área A e B", "E": "Área A e C"},
    "Para comparar as áreas, devemos somar as partes da área A e B, pois elas "
    "são superiores a C. Assim, a área A e B juntas têm 10 + 24 = 34 "
    "quadradinhos e a área C tem 36 quadradinhos. Portanto, a área C é a mais "
    "extensa.",
    "C", "HARD",
)

CAIXAS = _q(
    "Em uma biblioteca, há duas caixas de livros: uma é um retângulo de 6 cm por "
    "4 cm e a outra é um quadrado com 5 cm de lado. Qual das duas caixas tem "
    "maior área? A. A caixa retangular. B. A caixa quadrada. C. Ambas têm a "
    "mesma área. D. Não é possível comparar. E. A caixa de livros mais pesadas.",
    {"A": "A caixa retangular.", "B": "A caixa quadrada.",
     "C": "Ambas têm a mesma área.", "D": "Não é possível comparar.",
     "E": "A caixa de livros mais pesadas."},
    "Área da caixa retangular = 6 x 4 = 24 cm². Área da caixa quadrada = "
    "5 x 5 = 25 cm². Como 25 > 24, a caixa quadrada tem maior área.",
    "B", "HARD",
)


class TestFalsosPositivosDeComparacao(unittest.TestCase):
    """Problema A: a alternativa é um rótulo, não o valor — não pode reprovar."""

    def test_jardins_12x10(self):
        """diversidade_atual.json::ajustado::P08-5º-H16-N5::q4.

        _leading_number("Jardim com 12 x 10 m²") = 12; a resolução calcula 120 e
        112. O gabarito A está CERTO — reprovar era falso positivo puro.
        """
        self.assertEqual(check_consistency(JARDINS), (True, None))
        self.assertEqual(check_consistency_detalhado(JARDINS)[2], "corresponde")
        self.assertEqual(fix_gabarito(JARDINS)[1], "ok")

    def test_times_a_e_b(self):
        """diversidade_base_k0_s1.json::ajustado::P08-5º-H16-N5::q0.

        Alternativas literalmente "A"/"B": _leading_number devolve None em
        todas. É também o caso que valida a remoção do ARTIGO capitalizado —
        a frase final começa com "A área do time A é maior".
        """
        self.assertEqual(check_consistency(TIMES), (True, None))

    def test_areas_a_b_c(self):
        """diversidade_exp_C_s1.json::ajustado::P08-5º-H16-N5::q0.

        A única conta casada (10 + 24 = 34) é passo IRRELEVANTE; quem decide é a
        frase final. Exige distinguir "Área A" de "Área C" por letra maiúscula.
        """
        self.assertEqual(check_consistency(AREAS_ABC), (True, None))

    def test_caixas_retangular_vs_quadrada(self):
        """diversidade_exp_C_s1.json::ajustado::P08-5º-H16-N5::q3."""
        self.assertEqual(check_consistency(CAIXAS), (True, None))


class TestDiscriminacao(unittest.TestCase):
    """A regra nova não pode ser só permissiva: com o gabarito ERRADO, reprova."""

    def test_jardins_com_gabarito_trocado_reprova(self):
        q = dict(JARDINS, resposta_correta="B")
        self.assertEqual(check_consistency(q), (False, "A"))

    def test_caixas_com_gabarito_trocado_reprova(self):
        q = dict(CAIXAS, resposta_correta="A")
        self.assertEqual(check_consistency(q), (False, "B"))

    def test_areas_com_gabarito_trocado_reprova(self):
        q = dict(AREAS_ABC, resposta_correta="A")
        self.assertEqual(check_consistency(q), (False, "C"))


# ---------------------------------------------------------------------------
# Problema B: o resultado final não está em alternativa nenhuma.
# ---------------------------------------------------------------------------

PADARIA = _q(
    "Uma padaria registrou vendas por categoria no dia seguinte para o final do "
    "mês. Os dados estatísticos são: Pães: 120 unidades, Massas: 150 unidades, "
    "Biscoitos: 90 unidades, Cachorro-Quente: 80 unidades. Qual é a diferença "
    "entre o maior valor e o menor valor dessas categorias?",
    {"A": "60", "B": "100", "C": "110", "D": "120", "E": "130"},
    "O maior valor é 150 (Massas) e o menor valor é 80 (Cachorro-Quente). "
    "Calculando a diferença: 150 - 80 = 70. Portanto, a diferença é 70 unidades.",
    "C", "EASY",
)

PERIMETRO = _q(
    "Qual é o perímetro desse retângulo? Lados = 6 m e 8 m.",
    {"A": "14 m", "B": "24 m", "C": "12 m", "D": "48 m",
     "E": "Nenhuma das alternativas anteriores"},
    "Perímetro = 2 x (6 + 8) = 2 x 14 = 28 m.",
    "A", "HARD",
)

BALOES = _q(
    "Uma turma organizou uma festa. Os dados abaixo mostram a quantidade de "
    "balões: 12 balões azuis, 6 balões vermelhos e 24 balões brancos. Qual é a "
    "quantidade de balões vermelhos e brancos juntas?",
    {"A": "36 balões.", "B": "18 balões.", "C": "6 balões.", "D": "24 balões.",
     "E": "Nenhuma das alternativas anteriores"},
    "Soma dos balões vermelhos e brancos: 6 + 24 = 30 balões.",
    "A", "EASY",
)

RELOGIO = _q(
    "Um relógio analógico mostrava 13h00. Em quantos minutos esse horário já "
    "havia transcorrido?",
    {"A": "130 minutos", "B": "140 minutos", "C": "150 minutos",
     "D": "120 minutos", "E": "Nenhuma das alternativas anteriores"},
    "13h00 equivale a 13 x 60 = 780 minutos.",
    "A", "EASY",
)

FRACAO_EQUIVALENTE = _q(
    "Observe a fração 2/3. Qual das alternativas é uma fração equivalente a ela?",
    {"A": "4/12", "B": "3/9", "C": "4/8", "D": "6/12", "E": "5/13"},
    "Multiplicando numerador e denominador de 2/3 por 4: 2 x 4 = 8 e "
    "3 x 4 = 12, logo 8/12 = 2/3.",
    "D", "MEDIUM",
)

EQUACAO_5X = _q(
    "Em uma campanha de reciclagem, a equipe do 9º ano precisa calcular o valor "
    "inicial de uma dada expressão algébrica para distribuir recursos. A "
    "expressão é: 5x + 3 = 18. Sabendo que o valor final é 18 e que a operação "
    "final é somar 3, qual foi o valor inicial de x?",
    {"A": "4", "B": "5", "C": "6", "D": "7", "E": "8"},
    "Para descobrir o valor inicial x, devemos invertir a operação final. O "
    "valor final é 18 e a operação final é somar 3. Assim, subtraímos 3 de 18 "
    "para encontrar o valor inicial antes da operação: 18 - 3 = 15. Depois, "
    "usamos a expressão para encontrar x: 5x + 3 = 18 → 5x = 15 → x = 3. "
    "Portanto, o valor inicial de x é 3.",
    "C", "HARD",
)

FORA_DAS_ALTERNATIVAS = [
    ("padaria 150-80=70 fora de {60,100,110,120,130}", PADARIA),
    ("perímetro 28 fora de {14,24,12,48}", PERIMETRO),
    ("balões 30 fora de {36,18,6,24}", BALOES),
    ("relógio 780 fora de {130,140,150,120}", RELOGIO),
    ("fração 8/12 fora de {4/12,3/9,4/8,6/12,5/13}", FRACAO_EQUIVALENTE),
    ("equação 15 fora de {4,5,6,7,8}", EQUACAO_5X),
]


class TestRespostaForaDasAlternativas(unittest.TestCase):
    """Problema B: erro IRRECUPERÁVEL — não há letra para fix_gabarito sugerir."""

    def test_seis_casos_reais(self):
        for nome, questao in FORA_DAS_ALTERNATIVAS:
            with self.subTest(nome):
                ok, sugestao, motivo = check_consistency_detalhado(questao)
                self.assertIs(ok, False)
                self.assertIsNone(sugestao, "não pode sugerir letra nenhuma")
                self.assertEqual(motivo, MOTIVO_FORA_DAS_ALTERNATIVAS)
                self.assertTrue(resposta_fora_das_alternativas(questao))
                self.assertEqual(fix_gabarito(questao)[1], "fora_das_alternativas")

    def test_contrato_de_check_consistency_preservado(self):
        """check_consistency continua devolvendo uma tupla de 2 (ok, sugestao)."""
        for nome, questao in FORA_DAS_ALTERNATIVAS:
            with self.subTest(nome):
                self.assertEqual(check_consistency(questao), (False, None))

    def test_fix_gabarito_nao_reescreve_o_gabarito(self):
        """Não há letra certa; a questão sai INTACTA para ser regenerada."""
        for nome, questao in FORA_DAS_ALTERNATIVAS:
            with self.subTest(nome):
                corrigida, _status = fix_gabarito(questao)
                self.assertEqual(corrigida["resposta_correta"],
                                 questao["resposta_correta"])

    def test_nao_sugere_a_letra_nenhuma_das_anteriores(self):
        """PERIMETRO tem E = "Nenhuma das alternativas anteriores" e 28 não está
        nas demais. Ainda assim NÃO se sugere E: em 2 dos 5 casos reais com NDA
        a própria resolução está errada (malha 3x3 que calcula "3 x 1 = 3"), e
        trocar o gabarito para E publicaria uma questão errada de outro jeito.
        """
        self.assertEqual(check_consistency(PERIMETRO)[1], None)

    def test_gabarito_trocado_continua_recuperavel(self):
        """Quando o resultado ESTÁ em outra alternativa, o veredito é o antigo:
        (False, letra) e fix_gabarito conserta. É o que separa os dois casos."""
        q = dict(PADARIA, alternativas={"A": "60", "B": "70", "C": "110",
                                        "D": "120", "E": "130"})
        ok, sugestao, motivo = check_consistency_detalhado(q)
        self.assertEqual((ok, sugestao, motivo), (False, "B", "gabarito_errado"))
        self.assertFalse(resposta_fora_das_alternativas(q))
        corrigida, status = fix_gabarito(q)
        self.assertEqual(status, "corrigido")
        self.assertEqual(corrigida["resposta_correta"], "B")


class TestGabaritoNDANoRamoNumerico(unittest.TestCase):
    """Gabarito "Nenhuma das anteriores" e resultado ausente: NÃO verificável.

    Antes devolvia False (falso positivo silencioso). Não dá para aprovar —
    a conta da própria resolução pode estar errada — nem reprovar, porque
    "nenhuma" é literalmente o que a alternativa afirma.
    """

    def test_nda_correto_nao_e_reprovado(self):
        q = dict(PERIMETRO, resposta_correta="E")
        ok, sugestao, motivo = check_consistency_detalhado(q)
        self.assertIsNone(ok)
        self.assertIsNone(sugestao)
        self.assertEqual(motivo, "gabarito_nda")
        self.assertEqual(fix_gabarito(q)[1], "nao_verificavel")


# ---------------------------------------------------------------------------
# Classificador de alternativa — é ele, e não o enunciado, que discrimina.
# ---------------------------------------------------------------------------

class TestClasseAlternativa(unittest.TestCase):
    def test_valores_isolados(self):
        for txt in ("14 m", "36 balões.", "12 unidades de área",
                    "20 PÁGINAS.", "23,8%", "8.5", "R$ 12,50", "110"):
            with self.subTest(txt):
                self.assertEqual(classe_alternativa(txt), "numero")

    def test_unidade_nao_alfabetica_cai_no_lado_conservador(self):
        """"450π" tem o π fora de [a-zà-ÿ] e sai como textual. É a direção
        segura: a questão vai para o ramo de correspondência, que na pior das
        hipóteses devolve "não verificável" — nunca uma acusação falsa."""
        self.assertEqual(classe_alternativa("450π"), "textual")

    def test_fracoes(self):
        for txt in ("4/12", "6/12", "5/13"):
            with self.subTest(txt):
                self.assertEqual(classe_alternativa(txt), "fracao")

    def test_rotulos_e_frases_sao_textuais(self):
        """São estes que quebravam o verificador: número no MEIO ou ausente."""
        for txt in ("Área C", "A", "A caixa quadrada.", "Jardim com 12 x 10 m²",
                    "Placas de 4 placas; placas de 5 placas.", "Acutângulo",
                    "2 000 g"):
            with self.subTest(txt):
                self.assertEqual(classe_alternativa(txt), "textual")

    def test_nenhuma_das_anteriores(self):
        for txt in ("Nenhuma das alternativas anteriores", "N.D.A.",
                    "Nenhuma das opções", "Não é possível comparar."):
            with self.subTest(txt):
                self.assertEqual(classe_alternativa(txt), "nda")


class TestGuardaDeAlternativasDescritivas(unittest.TestCase):
    """"4 lados iguais" começa com dígito, mas o número é ADJETIVO, não resposta.

    12 das 918 questões do corpus têm >=2 alternativas assim. Exigir valores
    DISTINTOS entre as alternativas de valor impede que essas caiam no ramo
    numérico e virem acusação falsa de "resposta fora das alternativas".
    """

    def test_valores_repetidos_nao_entram_no_ramo_numerico(self):
        q = _q(
            "Qual característica define um quadrado?",
            {"A": "4 lados iguais", "B": "4 ângulos retos", "C": "4 vértices",
             "D": "2 diagonais", "E": "Nenhuma das alternativas anteriores"},
            "Um quadrado tem 2 + 2 = 4 pares de elementos congruentes.",
            "A",
        )
        ok, _sugestao, motivo = check_consistency_detalhado(q)
        self.assertIsNot(ok, False)
        self.assertNotEqual(motivo, MOTIVO_FORA_DAS_ALTERNATIVAS)


class TestSeparadorDeMilhar(unittest.TestCase):
    """Ponto de milhar pt-BR lido como decimal — 4 falsos positivos no train."""

    def test_soma_com_ponto_de_milhar(self):
        """data/train_curado.jsonl linha 33: "125.438 + 234.729 = 360.167"
        com alternativa "360167". A questão está CERTA."""
        q = _q(
            "Estádio recebeu 125.438 pessoas em campeonato. Ano seguinte: "
            "234.729. Quantas pessoas nesses dois anos juntos?",
            {"A": "360057", "B": "360167", "C": "360077", "D": "359967",
             "E": "Nenhuma das alternativas anteriores"},
            "A situação é aditiva: 125.438 + 234.729 = 360.167.",
            "B", "HARD",
        )
        self.assertEqual(check_consistency(q), (True, None))
        self.assertEqual(check_consistency_detalhado(q)[2], "valor_bate_milhar")

    def test_decimal_ascii_nao_e_confundido_com_milhar(self):
        """"0,875" arredondado e "8.5" continuam sendo decimais: o token de
        milhar exige grupos de 3 dígitos e parte inteira começando em 1-9."""
        q = _q("Média?", {"A": "8", "B": "8.5", "C": "9", "D": "8.25",
                          "E": "8.75"},
               "Soma: 8 + 9 = 17. Média: 17 / 2 = 8.5",
               "B")
        self.assertEqual(check_consistency(q), (True, None))


class TestCorrespondenciaTextual(unittest.TestCase):
    """Casos reais do train_curado que o ramo textual passa a aprovar."""

    def test_triangulo_retangulo_por_pitagoras(self):
        """train_curado linha 633: resultado 169 não é alternativa nenhuma
        ("Equilátero"/"Retângulo"/...); quem decide é a frase final."""
        q = _q(
            "Triângulo com ângulo de 90° e lados 5 cm, 12 cm e 13 cm. Esse "
            "triângulo é:",
            {"A": "Equilátero", "B": "Isósceles", "C": "Retângulo",
             "D": "Obtusângulo", "E": "Nenhuma das alternativas anteriores"},
            "Obedece Pitágoras: 5²+12²=25+144=169=13² → triângulo retângulo",
            "C", "HARD",
        )
        self.assertEqual(check_consistency(q), (True, None))

    def test_separador_de_frase_apos_digito(self):
        """O divisor ingênuo `(?<!\\d)[.;!?](?!\\d)` não corta "3x6=18. Área do
        time B", porque há dígito ANTES do ponto — e a "última frase" virava a
        resolução inteira, perdendo o caso dos times A/B."""
        self.assertEqual(check_consistency(TIMES), (True, None))

    def test_ambiguidade_vira_nao_verificavel_nunca_acusacao(self):
        """diversidade_candidato_unico.json::atual::P01-9º-H17-N10::q8.

        Perda de cobertura ACEITA conscientemente: hoje essa questão é marcada
        por ACIDENTE aritmético (3 + 4 = 7 não bate com "Nenhuma"), não porque
        a regra funcione. O erro é semântico ("não pode formar triângulo" é
        falso) e está fora do alcance de qualquer regex; quem deve pegá-lo é o
        gate de coerência resolução<->enunciado, não o de consistência numérica.
        """
        q = _q(
            "Triângulo com lados 3, 4, 6 — qual é o tipo de triângulo?",
            {"A": "Retângulo", "B": "Obtusângulo", "C": "Acutângulo",
             "D": "Isósceles", "E": "Nenhuma das alternativas anteriores"},
            "não pode formar triângulo (3 + 4 = 7 > 6; 3 + 6 > 4 e 4 + 6 > 3)",
            "E", "EASY",
        )
        self.assertEqual(check_consistency(q), (None, None))


class TestContratoPreservado(unittest.TestCase):
    """As duas primeiras verificações são as de sempre — zero regressão."""

    def test_valor_do_gabarito_bate_conta(self):
        q = _q("x", {"A": "12", "B": "13", "C": "18", "D": "19", "E": "20"},
               "Primeiro, 6 + 7 = 13. Depois, 13 + 5 = 18.", "C")
        self.assertEqual(check_consistency(q), (True, None))
        self.assertEqual(check_consistency_detalhado(q)[2], "valor_bate")

    def test_sem_conta_continua_nao_verificavel(self):
        q = _q("x", dict(zip("ABCDE", "abcde")),
               "É um triângulo obtusângulo.", "A")
        self.assertEqual(check_consistency(q), (None, None))
        self.assertEqual(check_consistency_detalhado(q)[2], "sem_conta")

    def test_entrada_invalida(self):
        self.assertEqual(check_consistency(None), (None, None))
        self.assertEqual(check_consistency({"resposta_correta": "A"}), (None, None))
        self.assertFalse(resposta_fora_das_alternativas(None))


if __name__ == "__main__":
    unittest.main()
