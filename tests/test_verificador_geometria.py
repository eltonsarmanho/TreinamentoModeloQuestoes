"""Testes do verificador simbólico de geometria (src/verificador_geometria.py).

ORIGEM DOS CASOS
  * R1/R2: as 20 questões de 9º H17 do teste real do usuário (2026-10-01),
    copiadas LITERALMENTE de outputs/testes_locais/teste_20261001_103202.json
    (R1) e teste_20261001_103625.json (R2) — outputs/ é gitignored, por isso a
    cópia mora aqui. O rótulo de cada uma vem da auditoria humana
    (outputs/testes_locais/Log.txt): só 5 das 20 estavam corretas.
  * TREINO_REAL: itens do banco real em data/train_curado.jsonl (também
    gitignored). São a referência de FALSO POSITIVO: questões boas que o
    verificador não pode reprovar.
  * Casos-limite e "sem falso positivo": escritos à mão para travar cada regra
    e cada defeito achado durante a implementação (o docstring diz qual).

A regra de ouro do módulo é "na dúvida, nao_aplicavel": os testes de falso
positivo valem tanto quanto os de detecção.
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import verificador_geometria as vg  # noqa: E402

NDA = "Nenhuma das alternativas anteriores"
REJEITA = vg.GEO_REJEITA


def _q(enunciado, alternativas, gabarito, resolucao="", difficulty="EASY"):
    return {
        "enunciado": enunciado,
        "alternativas": dict(zip("ABCDE", alternativas)),
        "resolucao_passo_a_passo": resolucao,
        "resposta_correta": gabarito,
        "difficulty": difficulty,
    }


TRI_LADOS = ["Isósceles", "Escaleno", "Equilátero", "Retângulo", NDA]
TRI_ANGULOS = ["Acutângulo", "Retângulo", "Obtusângulo", "Equilátero", NDA]
QUAD = ["Retângulo", "Losango", "Quadrado", "Trapézio", "Paralelogramo"]


# ---------------------------------------------------------------------------
# As 20 questões auditadas (cópia literal; ver docstring do módulo).
# ---------------------------------------------------------------------------
R1 = [  # outputs/testes_locais/teste_20261001_103202.json, na ordem do arquivo (Q1..Q10)
    {
        "enunciado": (
            "Em uma festa junina, as barracas formam triângulos. A barraca A "
            "tem todos os ângulos agudos, a barraca B tem um ângulo obtuso e a "
            "barraca C tem um ângulo reto. Como os triângulos podem ser "
            "classificados quanto aos ângulos?"),
        "alternativas": {
            "A": (
                "A barraca A é um triângulo retângulo, a barraca B é um triângulo "
                "agudo e a barraca C é um triângulo obtusângulo."),
            "B": (
                "A barraca A é um triângulo obtusângulo, a barraca B é um "
                "triângulo retângulo e a barraca C é um triângulo agudo."),
            "C": (
                "A barraca A é um triângulo agudo, a barraca B é um triângulo "
                "retângulo e a barraca C é um triângulo obtusângulo."),
            "D": (
                "A barraca A é um triângulo agudo, a barraca B é um triângulo "
                "obtusângulo e a barraca C é um triângulo retângulo."),
            "E": (
                "A barraca A é um triângulo obtusângulo, a barraca B é um "
                "triângulo agudo e a barraca C é um triângulo retângulo."),
        },
        "resolucao_passo_a_passo": (
            "Triângulo com todos ângulos agudos → triângulo agudo. Triângulo "
            "com um ângulo obtuso → triângulo obtusângulo. Triângulo com um "
            "ângulo reto → triângulo retângulo."),
        "resposta_correta": "D",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Um ônibus transporta passageiros e, no trajeto, passa por quatro "
            "pontos formando um quadrilátero. Qual das alternativas mostra "
            "corretamente a classificação quanto aos lados desse quadrilátero?"),
        "alternativas": {
            "A": "Um losango",
            "B": "Um retângulo",
            "C": "Um triângulo",
            "D": "Um trapézio",
            "E": "Um quadrado",
        },
        "resolucao_passo_a_passo": "Quadrilátero com todos os lados iguais → losango.",
        "resposta_correta": "A",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma horta comunitária, há um jardim dividido em três áreas. "
            "Uma delas é um triângulo cujos lados medem 5 m, 7 m e 9 m. Qual "
            "tipo de triângulo essa área representa?"),
        "alternativas": {
            "A": "Retângulo",
            "B": "Acutângulo",
            "C": "Obtusângulo",
            "D": "Equilátero",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Os lados 5, 7 e 9 são todos diferentes, logo é um triângulo "
            "acutângulo."),
        "resposta_correta": "B",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma biblioteca, há um espelho formando um quadrilátero. Qual "
            "das seguintes descrições corresponde ao espelho, considerando "
            "apenas os ângulos? Os ângulos do espelho são todos obtusos."),
        "alternativas": {
            "A": "Quadrado",
            "B": "Retângulo",
            "C": "Losango",
            "D": "Trapézio",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Os ângulos do losango são todos obtusos (ângulos de 120°), "
            "enquanto os outros tipos de quadrilátero possuem ângulos retos "
            "(quadrado), acutos (trapézio) ou obtusos (retângulo)."),
        "resposta_correta": "C",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Durante a reforma de uma casa, o pedreiro precisa classificar "
            "triângulos formados com os seguintes ângulos: um triângulo com "
            "dois ângulos retos e um triângulo com todos os ângulos agudos. "
            "Como esses triângulos são chamados?"),
        "alternativas": {
            "A": "Obtusângulo",
            "B": "Eclipse",
            "C": "Retângulo",
            "D": "Acutângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Triângulo com dois ângulos retos é retângulo. Triângulo com "
            "ângulos agudos é acutângulo."),
        "resposta_correta": "D",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Durante uma campanha de reciclagem, Ana coloca quatro peças em um "
            "canteiro para formar um polígono. As peças têm as seguintes "
            "medidas: 5 cm, 7 cm, 8 cm e 10 cm. A forma do polígono formado "
            "pode ser classificada como:"),
        "alternativas": {
            "A": "Triângulo",
            "B": "Retângulo",
            "C": "Losango",
            "D": "Trapézio",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Os quatro lados têm medidas diferentes, então o polígono é um "
            "quadrilátero qualquer, e não pode ser classificado como "
            "triângulo, retângulo, losango ou trapézio."),
        "resposta_correta": "E",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Triângulo tem lados de 5 cm, 5 cm e 8 cm. Qual tipo de triângulo "
            "esse é?"),
        "alternativas": {
            "A": "Isósceles",
            "B": "Escaleno",
            "C": "Equilátero",
            "D": "Retângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Triângulo possui dois lados iguais (5 cm e 5 cm) e um diferente "
            "(8 cm). Logo, é um triângulo isósceles."),
        "resposta_correta": "A",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma receita de bolo, uma fábrica usou uma caixa de formato "
            "quadrilátero para armazenar ingredientes. Para evitar que o bolo "
            "ficasse de pé, a fábrica decidiu cobrir a caixa com uma tampa. "
            "Qual dos seguintes tipos de quadrilátero pode ser coberto com "
            "essa tampa, sem que o ângulo interior se sobreponha?"),
        "alternativas": {
            "A": "Retângulo",
            "B": "Trapézio",
            "C": "Quadrado",
            "D": "Losango",
            "E": "Triângulo",
        },
        "resolucao_passo_a_passo": (
            "Um quadrado tem quatro ângulos retos (90°), o que é compatível "
            "com a cobertura de uma tampa sem sobreposição. Já o losango tem "
            "dois pares de ângulos congruentes (mas não necessariamente todos "
            "retos), o trapézio tem três ângulos retos (se for um retângulo) e "
            "o triângulo não tem ângulo reto. Portanto, o único quadrilátero "
            "com todos os ângulos retos é o quadrado."),
        "resposta_correta": "C",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Triângulo com dois ângulos obtusos. Como ele é classificado "
            "quanto aos ângulos?"),
        "alternativas": {
            "A": "Escaleno",
            "B": "Obtusângulo",
            "C": "Retângulo",
            "D": "Acutângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": "Triângulo com dois ângulos obtusos é chamado de obtusângulo",
        "resposta_correta": "B",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Na loja de brinquedos, as prateleiras têm as seguintes formas: "
            "uma é um quadrilátero com todos os lados iguais e ângulos retos; "
            "outra é um trapézio com três lados iguais e um lado diferente; a "
            "terceira é um quadrilátero com dois lados iguais e dois lados "
            "diferentes; a quarta é um losango; e a quinta é um triângulo. Em "
            "qual dessas prateleiras as formas dos lados correspondem ao tipo "
            "de quadrilátero descrito?"),
        "alternativas": {
            "A": "Triângulo",
            "B": "Quadrilátero com dois lados iguais e dois lados diferentes",
            "C": "Quadrilátero com todos os lados iguais",
            "D": "Trapézio com três lados iguais",
            "E": "Losango",
        },
        "resolucao_passo_a_passo": (
            "A prateleira que tem um quadrilátero com todos os lados iguais é "
            "o losango. Os outros tipos não se encaixam nesse formato."),
        "resposta_correta": "E",
        "difficulty": "EASY",
    },
]

R2 = [  # outputs/testes_locais/teste_20261001_103625.json, na ordem do arquivo (Q1..Q10)
    {
        "enunciado": (
            "Na festa junina, os organizadores decoraram uma barraca "
            "triangular com as seguintes medidas: lado A = 8 m, lado B = 10 m "
            "e lado C = 12 m. Como os lados são diferentes entre si, qual é o "
            "tipo de triângulo formado?"),
        "alternativas": {
            "A": "Isósceles",
            "B": "Retângulo",
            "C": "Equilátero",
            "D": "Obtusângulo",
            "E": "Escaleno",
        },
        "resolucao_passo_a_passo": (
            "Triângulo escaleno possui todos os lados diferentes, como em um "
            "triângulo com lados 8 m, 10 m e 12 m."),
        "resposta_correta": "E",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma loja de brinquedos, o carrinho de brinquedos tem quatro "
            "lados com os seguintes ângulos: 90°, 90°, 90° e 90°. Como ele é "
            "classificado?"),
        "alternativas": {
            "A": "Losango",
            "B": "Retângulo",
            "C": "Quadrado",
            "D": "Trapézio",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": "Carrinho com quatro ângulos retos ⇒ é um quadrado",
        "resposta_correta": "C",
        "difficulty": "EASY",
    },
    {
        "enunciado": "Triângulo com ângulos: 90°, 45°, 45°. Qual tipo de triângulo?",
        "alternativas": {
            "A": "Retângulo",
            "B": "Obtusângulo",
            "C": "Isósceles",
            "D": "Escaleno",
            "E": "Equilátero",
        },
        "resolucao_passo_a_passo": (
            "Triângulo com um ângulo de 90° é retângulo. Triângulo com dois "
            "ângulos de 45° é isósceles."),
        "resposta_correta": "A",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma horta comunitária, o canteiro A tem quatro lados de mesmo "
            "comprimento e o canteiro B tem lados de 5 cm, 5 cm e 10 cm. Quais "
            "desses canteiros são retângulos?"),
        "alternativas": {
            "A": "Canteiro B",
            "B": "Canteiro A",
            "C": "Ambos são retângulos",
            "D": "Nenhum dos canteiros é retângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Canteiro A tem todos os lados iguais, portanto é um quadrado, que "
            "é um tipo de retângulo. Canteiro B tem lados 5,5 e 10 cm, logo "
            "não é retângulo. Portanto, apenas Canteiro A é retângulo."),
        "resposta_correta": "B",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Na biblioteca da escola, há um conjunto de prateleias formadas em "
            "triângulos. Um desses triângulos tem todos os lados iguais. Como "
            "se chama esse tipo de triângulo?"),
        "alternativas": {
            "A": "Escaleno",
            "B": "Isósceles",
            "C": "Retângulo",
            "D": "Equilátero",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": "Triângulo com todos os lados iguais é chamado de equilátero",
        "resposta_correta": "D",
        "difficulty": "EASY",
    },
    {
        "enunciado": "Qual das alternativas mostra um quadrilátero com dois ângulos obtusos?",
        "alternativas": {
            "A": "Retângulo",
            "B": "Losango",
            "C": "Trapézio",
            "D": "Quadrado",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Retângulo e quadrado têm ângulos retos; trapézio pode ter dois "
            "ângulos obtusos (como trapézio isósceles); losango tem ângulos "
            "opostos iguais, podendo ser obtusos. Portanto, o trapézio (C) é o "
            "único com dois ângulos obtusos."),
        "resposta_correta": "C",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Em uma campanha de reciclagem, três garrafas PET foram separadas "
            "e suas bases formaram um triângulo. Qual das alternativas "
            "representa corretamente o tipo de triângulo formado?"),
        "alternativas": {
            "A": "Equilátero",
            "B": "Obtusângulo",
            "C": "Retângulo",
            "D": "Acrisângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "A soma dos ângulos internos de qualquer triângulo é 180°. Nenhuma "
            "das alternativas menciona a soma de ângulos ou os lados, apenas "
            "os ângulos. Assim, o triângulo é classificado como 'Nenhuma das "
            "alternativas anteriores'."),
        "resposta_correta": "E",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Um cofrinho tem as seguintes dimensões: 3 cm, 4 cm e 5 cm de "
            "lado. Se o cofrinho for redimensionado para ter os mesmos lados, "
            "qual é o tipo de quadrilátero formado?"),
        "alternativas": {
            "A": "Retângulo",
            "B": "Trapézio",
            "C": "Losango",
            "D": "Quadrado",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": "Os lados são iguais (3 cm = 4 cm = 5 cm), logo é um quadrado.",
        "resposta_correta": "D",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Triângulo ABC tem lados AB=5 cm, BC=5 cm e CA=6 cm. Qual é o tipo "
            "de triângulo? A) Escaleno B) Isósceles C) Equilátero D) Retângulo "
            "E) Nenhuma das alternativas anteriores"),
        "alternativas": {
            "A": "Escaleno",
            "B": "Isósceles",
            "C": "Equilátero",
            "D": "Retângulo",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": "Triângulo tem dois lados iguais (AB=BC=5 cm) → Isósceles.",
        "resposta_correta": "B",
        "difficulty": "EASY",
    },
    {
        "enunciado": (
            "Na receita de bolo, o biscoito tem os seguintes ângulos: 90°, "
            "90°, 90° e 90°. Como o biscoito é feito de um material que "
            "permite flexibilidade, qual tipo de quadrilátero ele é?"),
        "alternativas": {
            "A": "Quadrado",
            "B": "Losango",
            "C": "Retângulo",
            "D": "Trapézio",
            "E": "Nenhuma das alternativas anteriores",
        },
        "resolucao_passo_a_passo": (
            "Um quadrilátero com quatro ângulos retos é um quadrado, pois "
            "todos os lados são iguais e os ângulos são iguais. Portanto, o "
            "biscoito é um quadrado."),
        "resposta_correta": "A",
        "difficulty": "EASY",
    },
]

AUDITADAS = {f"R1-Q{i}": q for i, q in enumerate(R1, 1)}
AUDITADAS.update({f"R2-Q{i}": q for i, q in enumerate(R2, 1)})

# Rótulos da auditoria humana (Log.txt).
APROVAR = {"R1-Q1", "R1-Q7", "R2-Q1", "R2-Q5", "R2-Q9"}

# Veredito exato que o verificador dá a cada reprovada, com o porquê. O critério
# de aceitação é só "reprova" (qualquer veredito de GEO_REJEITA); fixar o
# veredito exato documenta o raciocínio e pega regressão silenciosa.
ESPERADO = {
    # nada sobre os lados: "passa por quatro pontos formando um quadrilátero"
    "R1-Q2": ("dados_insuficientes", None),
    # 25 + 49 = 74 < 81: obtusângulo (C), não acutângulo
    "R1-Q3": ("gabarito_errado", "C"),
    # os 4 ângulos de um quadrilátero não podem ser todos obtusos
    "R1-Q4": ("premissa_impossivel", None),
    # "um triângulo com dois ângulos retos" (primeira das duas figuras)
    "R1-Q5": ("premissa_impossivel", None),
    # 5, 7, 8, 10: trapézio é possível, então "nenhuma" não é garantida
    "R1-Q6": ("dados_insuficientes", None),
    # nenhum dado sobre a caixa; quadrado não é garantido
    "R1-Q8": ("dados_insuficientes", None),
    # dois ângulos obtusos num triângulo
    "R1-Q9": ("premissa_impossivel", None),
    # C "Quadrilátero com todos os lados iguais" ≡ E "Losango"
    "R1-Q10": ("nao_unica", None),
    # 4 ângulos de 90°: retângulo (B) garantido, quadrado não
    "R2-Q2": ("gabarito_errado", "B"),
    # 90-45-45: retângulo E isósceles, eixos diferentes
    "R2-Q3": ("nao_unica", None),
    # canteiro B: 5 + 5 = 10, triângulo degenerado
    "R2-Q4": ("premissa_impossivel", None),
    # losango e trapézio podem ter dois ângulos obtusos
    "R2-Q6": ("nao_unica", None),
    # garrafas PET: nenhuma medida (a auditoria aceita dados_insuficientes)
    "R2-Q7": ("dados_insuficientes", None),
    # "3 cm, 4 cm e 5 cm de lado" num quadrilátero: quadrado é impossível
    "R2-Q8": ("gabarito_errado", None),
    # 4 ângulos de 90°: retângulo (C) garantido, quadrado não
    "R2-Q10": ("gabarito_errado", "C"),
}


class TestAuditadas(unittest.TestCase):
    """Critério de aceitação: aprovar as 5 corretas e reprovar as 15 erradas."""

    def test_rotulos_completos(self):
        self.assertEqual(len(AUDITADAS), 20)
        self.assertEqual(set(AUDITADAS), APROVAR | set(ESPERADO))

    def test_aprova_as_cinco_corretas(self):
        for nome in sorted(APROVAR):
            with self.subTest(nome):
                v, det = vg.verificar_geometria(AUDITADAS[nome])
                self.assertEqual(v, "ok", det)

    def test_reprova_as_quinze_erradas(self):
        for nome in sorted(ESPERADO):
            with self.subTest(nome):
                v, det = vg.verificar_geometria(AUDITADAS[nome])
                self.assertIn(v, REJEITA, det)

    def test_veredito_e_sugestao_de_cada_reprovada(self):
        for nome, (veredito, sugestao) in sorted(ESPERADO.items()):
            with self.subTest(nome):
                v, det = vg.verificar_geometria(AUDITADAS[nome])
                self.assertEqual(v, veredito, det)
                self.assertEqual(det.get("sugestao"), sugestao, det)

    def test_detalhe_explica_o_veredito(self):
        """O detalhe traz fatos, valor V/F/I de cada alternativa e motivo legível."""
        v, det = vg.verificar_geometria(AUDITADAS["R1-Q3"])
        self.assertEqual(det["valores"], {"A": "F", "B": "F", "C": "V", "D": "F", "E": "F"})
        self.assertIn("lados: 5, 7, 9", det["fatos"])
        self.assertTrue(det["explicacao"])
        self.assertTrue(det["leituras_concordam"])

    @unittest.skipUnless((ROOT / "outputs" / "testes_locais").exists(), "outputs/ ausente")
    def test_copia_literal_confere_com_os_arquivos(self):
        base = ROOT / "outputs" / "testes_locais"
        for nome, copia in (("teste_20261001_103202.json", R1), ("teste_20261001_103625.json", R2)):
            caminho = base / nome
            if not caminho.exists():
                self.skipTest(f"{nome} ausente")
            original = json.loads(caminho.read_text(encoding="utf-8"))
            self.assertEqual(original["resultados"][0]["questoes"], copia)


class TestUniverso(unittest.TestCase):
    """Os 41 estados existem (testemunha) e as impossibilidades conhecidas faltam."""

    def test_cada_testemunha_reproduz_seu_estado(self):
        chaves = ("fig", "part", "r", "o", "a", "npar", "opp", "pipa", "pernas_iguais")
        for i, estado in enumerate(vg.UNIVERSO):
            with self.subTest(i=i):
                if estado["fig"] == "tri":
                    calc = vg.caracteristicas_triangulo(estado["testemunha"])
                else:
                    calc = vg.caracteristicas_quadrilatero(estado["testemunha"])
                self.assertIsNotNone(calc)
                self.assertEqual({k: calc[k] for k in chaves}, {k: estado[k] for k in chaves})

    def test_contagem_e_estados_distintos(self):
        self.assertEqual(len(vg.UNIVERSO), 41)
        self.assertEqual(len(vg.TRI), 7)
        self.assertEqual(len(vg.QUAD), 34)
        chaves = ("fig", "part", "r", "o", "a", "npar", "opp", "pipa", "pernas_iguais")
        assinaturas = {tuple(e[k] for k in chaves) for e in vg.UNIVERSO}
        self.assertEqual(len(assinaturas), 41)

    def test_impossibilidades_classicas_nao_estao_no_universo(self):
        for e in vg.UNIVERSO:
            with self.subTest(e=e["testemunha"]):
                self.assertEqual(e["r"] + e["o"] + e["a"], 3 if e["fig"] == "tri" else 4)
                if e["fig"] == "tri":
                    self.assertLessEqual(e["r"] + e["o"], 1)       # no máx. 1 reto/obtuso
                    if e["part"] == (3,):
                        self.assertEqual(e["a"], 3)                # equilátero é acutângulo
                else:
                    self.assertNotEqual(e["r"], 3)                 # 3 retos => o 4º é reto
                    self.assertNotIn(e["o"], (4,))                 # 4 obtusos somam > 360°
                    self.assertNotEqual(e["a"], 4)                 # 4 agudos somam < 360°
                    if e["npar"] == 1:                             # trapézio
                        self.assertNotIn(e["part"], ((4,), (2, 2)))
                        self.assertIn((e["r"], e["o"], e["a"]), ((0, 2, 2), (2, 1, 1)))
                    if e["opp"] == 2:                              # lados opostos iguais
                        self.assertEqual(e["npar"], 2)

    def test_caracteristicas_recusam_figuras_degeneradas(self):
        self.assertIsNone(vg.caracteristicas_triangulo((1, 4, 9)))   # 1 + 2 = 3
        self.assertIsNone(vg.caracteristicas_triangulo((4, 4, 16)))  # 2 + 2 = 4
        self.assertIsNone(vg.caracteristicas_quadrilatero(((0, 0), (1, 0), (2, 0), (0, 1))))
        self.assertIsNone(vg.caracteristicas_quadrilatero(((0, 0), (2, 0), (1, 1), (1, 3))))


class TestHierarquia(unittest.TestCase):
    """Convenções 1-3 do módulo, como inclusão de conjuntos."""

    def test_quadrilateros_inclusivos_e_trapezio_exclusivo(self):
        for L in vg.LEITURAS:
            R = vg.ROTULOS[L]
            with self.subTest(leitura=L):
                self.assertLess(R["quadrado"], R["retangulo_q"])
                self.assertLess(R["retangulo_q"], R["paralelogramo"])
                self.assertLess(R["quadrado"], R["losango"])
                self.assertLess(R["losango"], R["paralelogramo"])
                self.assertEqual(R["quadrado"], R["retangulo_q"] & R["losango"])
                self.assertFalse(R["trapezio"] & R["paralelogramo"])
                self.assertLess(R["trapezio_isosceles"], R["trapezio"])
                self.assertLess(R["trapezio_retangulo"], R["trapezio"])

    def test_isosceles_inclusivo_na_leitura_A_exclusivo_na_B(self):
        A, B = vg.ROTULOS["A"], vg.ROTULOS["B"]
        self.assertLess(A["equilatero"], A["isosceles"])
        self.assertFalse(B["equilatero"] & B["isosceles"])


class TestCasosLimite(unittest.TestCase):
    def assertVeredito(self, q, esperado, sugestao="qualquer"):
        v, det = vg.verificar_geometria(q)
        self.assertEqual(v, esperado, det)
        if sugestao != "qualquer":
            self.assertEqual(det.get("sugestao"), sugestao, det)
        return det

    # --- desigualdade triangular ESTRITA (degenerado é impossível)
    def test_degenerados_sao_premissa_impossivel(self):
        for lados in ("5, 5 e 10", "2 cm, 2 cm e 4 cm", "4, 4 e 8"):
            with self.subTest(lados=lados):
                self.assertVeredito(_q(f"Um triângulo tem lados {lados}. Qual é o tipo quanto aos "
                                       "lados?", TRI_LADOS, "A"), "premissa_impossivel")

    # --- soma dos ângulos
    def test_soma_errada_e_premissa_impossivel(self):
        self.assertVeredito(_q("Um triângulo tem ângulos de 100°, 50° e 40°. Qual é o tipo?",
                               TRI_ANGULOS, "C"), "premissa_impossivel")
        # caso real G-9H17-0110: 90 + 3 x 120 = 450
        self.assertVeredito(_q("Uma sala tem formato de quadrilátero e, em seu canto, há um ângulo "
                               "de 90 graus. Os outros ângulos são de 120 graus cada um. Como ele "
                               "pode classificar esse quadrilátero?", QUAD, "D"),
                            "premissa_impossivel")

    def test_soma_180_e_teorema_nao_dado(self):
        self.assertVeredito(_q("Os ângulos internos de um triângulo somam 180°. Um triângulo tem "
                               "ângulos de 30°, 60° e 90°. Classifique-o quanto aos ângulos.",
                               TRI_ANGULOS, "B"), "ok")

    def test_terceiro_angulo_e_calculado(self):
        self.assertVeredito(_q("Dois ângulos de um triângulo medem 30° e 60°. Classifique-o "
                               "quanto aos ângulos.", TRI_ANGULOS, "B"), "ok")

    def test_maior_que_90_e_obtuso_nao_90(self):
        self.assertVeredito(_q("Um triângulo tem um ângulo maior que 90°. Classifique-o quanto "
                               "aos ângulos.", TRI_ANGULOS, "C"), "ok")
        self.assertVeredito(_q("Um triângulo tem todos os ângulos menores que 90°. Classifique-o "
                               "quanto aos ângulos.", TRI_ANGULOS, "A"), "ok")

    def test_angulos_impossiveis_no_quadrilatero(self):
        for texto in ("três ângulos retos e um ângulo agudo", "todos os ângulos agudos",
                      "todos os ângulos obtusos"):
            with self.subTest(texto=texto):
                self.assertVeredito(_q(f"Um quadrilátero tem {texto}. Qual é o nome?", QUAD, "A"),
                                    "premissa_impossivel")
        # três obtusos é possível (pipa, por exemplo): nenhuma das 4 classes serve
        self.assertVeredito(_q("Um quadrilátero tem três ângulos obtusos. Qual é o nome?",
                               ["Retângulo", "Losango", "Quadrado", "Trapézio", NDA], "E"), "ok")

    def test_quadrilatero_concavo_fica_fora(self):
        self.assertVeredito(_q("Um quadrilátero tem ângulos de 200°, 60°, 50° e 50°. Qual é o "
                               "nome?", QUAD, "E"), "nao_aplicavel")

    # --- recíproca de Pitágoras (aritmética exata)
    def test_reciproca_de_pitagoras(self):
        casos = (("5 cm, 7 cm e 9 cm", "C"), ("3 cm, 4 cm e 6 cm", "C"), ("3 cm, 4 cm e 5 cm", "B"),
                 ("4 cm, 5 cm e 6 cm", "A"))
        for lados, gab in casos:
            with self.subTest(lados=lados):
                self.assertVeredito(_q(f"Um triângulo tem lados de {lados}. Classifique-o quanto "
                                       "aos ângulos.", TRI_ANGULOS, gab), "ok")

    # --- hierarquia dos quadriláteros
    def test_quatro_retos_garante_retangulo_nao_quadrado(self):
        en = "Um quadrilátero tem quatro ângulos retos. Qual é o nome?"
        self.assertVeredito(_q(en, QUAD, "A"), "ok")  # paralelogramo V, mas menos específico
        self.assertVeredito(_q(en, QUAD, "C"), "gabarito_errado", sugestao="A")

    def test_quatro_lados_iguais_garante_losango_nao_quadrado(self):
        en = "Um quadrilátero tem quatro lados iguais. Qual é o nome?"
        self.assertVeredito(_q(en, QUAD, "B"), "ok")
        self.assertVeredito(_q(en, QUAD, "C"), "gabarito_errado", sugestao="B")

    def test_quadrado_domina_as_superclasses(self):
        self.assertVeredito(_q("Um quadrilátero tem quatro lados iguais e quatro ângulos retos. "
                               "Qual é o nome?", QUAD, "C"), "ok")

    def test_quadrilatero_com_tres_medidas_nao_e_quadrado(self):
        self.assertVeredito(_q("Um quadrilátero tem lados de 3 cm, 4 cm e 5 cm. Qual é o nome?",
                               QUAD, "C"), "gabarito_errado")

    def test_retangulo_de_lados_4_e_6_nao_e_contradicao(self):
        self.assertVeredito(_q("Um retângulo tem lados de 4 cm e 6 cm. Qual é a classificação "
                               "mais específica?", ["Quadrado", "Retângulo", "Losango", "Trapézio",
                                                    NDA], "B"), "ok")

    # --- equilátero x isósceles (duas leituras)
    def test_equilatero_e_isosceles(self):
        en = "Um triângulo tem os três lados iguais. Como ele é classificado quanto aos lados?"
        # Equilátero listado: gabarito "Isósceles" não é o mais específico
        v, _ = vg.verificar_geometria(_q(en, ["Isósceles", "Escaleno", "Equilátero",
                                              "Obtusângulo", NDA], "A"))
        self.assertIn(v, REJEITA)
        # Equilátero listado e gabarito: Isósceles verdadeiro não atrapalha (R2-Q5)
        self.assertVeredito(_q(en, ["Isósceles", "Escaleno", "Equilátero", "Obtusângulo", NDA],
                               "C"), "ok")
        # sem Equilátero na lista: depende da convenção -> não opina
        self.assertVeredito(_q(en, ["Isósceles", "Escaleno", "Retângulo", "Obtusângulo", NDA],
                               "A"), "nao_aplicavel")

    # --- eixos (lados x ângulos)
    def test_eixos_cruzados_sem_eixo_nomeado_sao_nao_unica(self):
        casos = (("Triângulo com ângulos 90°, 45° e 45°. Qual é o tipo?",
                  ["Retângulo", "Obtusângulo", "Isósceles", "Escaleno", "Equilátero"], "A"),
                 ("Um triângulo tem ângulos de 60°, 60° e 60°. Qual é o tipo?",
                  ["Equilátero", "Obtusângulo", "Retângulo", "Acutângulo", NDA], "A"),
                 ("Triângulo com lados 7, 7 e 8. Qual é o tipo?",
                  ["Retângulo", "Obtusângulo", "Acutângulo", "Isósceles", NDA], "D"))
        for en, alts, gab in casos:
            with self.subTest(en=en):
                self.assertVeredito(_q(en, alts, gab), "nao_unica")

    def test_eixo_nomeado_desempata(self):
        # caso real G-9H17-0021: "classifique pelo ângulo interno" com gabarito Equilátero
        self.assertVeredito(_q("Triângulo com todos os lados iguais: classifique pelo ângulo "
                               "interno.", ["Retângulo", "Triângulo equilátero", "Triângulo escaleno",
                                            "Triângulo acutângulo", NDA], "B"),
                            "gabarito_errado", sugestao="D")
        self.assertVeredito(_q("Triângulo com ângulos 90°, 45° e 45°. Classifique-o quanto aos "
                               "ângulos.", ["Retângulo", "Obtusângulo", "Isósceles", "Escaleno",
                                            "Equilátero"], "A"), "ok")

    def test_deteccao_do_eixo(self):
        casos = {
            "como ele eh classificado quanto aos lados?": "lados",
            "classifique-o em relacao aos angulos internos.": "angulos",
            "classifique pelo angulo interno.": "angulos",
            "qual descricao corresponde, considerando apenas os angulos?": "angulos",
            "quanto a classificacao dos angulos, ele eh:": "angulos",
            "classifique-o quanto aos lados e aos angulos.": None,
            "qual eh o tipo de triangulo?": None,
            # dado, não eixo: "considerando os lados de 3, 4 e 5 cm"
            "considerando os lados de 3, 4 e 5 cm, como ele eh quanto aos angulos?": "angulos",
        }
        for pergunta, eixo in casos.items():
            with self.subTest(pergunta=pergunta):
                self.assertEqual(vg._eixo(pergunta), eixo)

    def test_lados_e_angulos_juntos_anulam_o_eixo(self):
        self.assertVeredito(_q("Triângulo com dois lados iguais e um ângulo de 90°. Classifique-o "
                               "quanto aos lados e aos ângulos.",
                               ["Isósceles e acutângulo", "Escaleno e retângulo",
                                "Isósceles e retângulo", "Equilátero e retângulo", NDA], "C"), "ok")

    # --- "Nenhuma das alternativas anteriores" e "não é possível determinar"
    def test_nda_so_vale_com_todas_as_outras_falsas(self):
        en = "Um triângulo tem lados de 3 cm, 4 cm e 5 cm. Como ele é classificado quanto aos lados?"
        self.assertVeredito(_q(en, ["Isósceles", "Equilátero", "Acutângulo", "Obtusângulo", NDA],
                               "E"), "ok")
        self.assertVeredito(_q(en, ["Isósceles", "Equilátero", "Escaleno", "Obtusângulo", NDA],
                               "E"), "gabarito_errado", sugestao="C")

    def test_nao_e_possivel_determinar(self):
        self.assertVeredito(_q("Um quadrilátero tem dois ângulos obtusos e dois agudos. Qual é o "
                               "nome?", ["Retângulo", "Quadrado", "Losango", "Trapézio",
                                         "Não é possível determinar"], "E"), "ok")

    # --- alternativas equivalentes
    def test_alternativas_equivalentes(self):
        self.assertVeredito(_q("Um quadrilátero tem quatro lados iguais. Qual é o nome?",
                               ["Quadrilátero com todos os lados iguais", "Losango", "Retângulo",
                                "Trapézio", NDA], "B"), "nao_unica")
        # caso real G-9H17-0025: "um ângulo obtuso" ≡ "um obtuso e dois agudos"
        self.assertVeredito(_q("Qual dos seguintes triângulos é classificado como obtusângulo?",
                               ["Triângulo com dois lados iguais.",
                                "Triângulo com todos os ângulos obtusos.",
                                "Triângulo com um ângulo obtuso e dois ângulos agudos.",
                                "Triângulo com um ângulo obtuso.", NDA], "D"), "nao_unica")

    # --- modo reverso (a propriedade está na pergunta)
    def test_modo_reverso(self):
        alts = ["Retângulo", "Losango", "Trapézio", "Quadrado", NDA]
        self.assertVeredito(_q("Qual das alternativas mostra um quadrilátero com dois ângulos "
                               "obtusos?", alts, "C"), "nao_unica")
        self.assertVeredito(_q("Qual quadrilátero pode ter dois ângulos obtusos?",
                               ["Retângulo", "Quadrado", "Trapézio", "Triângulo", NDA], "C"), "ok")
        self.assertVeredito(_q("Qual triângulo tem um ângulo obtuso?",
                               ["Triângulo equilátero", "Triângulo obtusângulo",
                                "Triângulo retângulo", "Triângulo acutângulo", NDA], "B"), "ok")

    def test_dados_na_frase_interrogativa_sao_modo_direto(self):
        self.assertVeredito(_q("Triângulo com ângulos 80°, 60° e 40° — qual é o tipo?",
                               TRI_ANGULOS, "A"), "ok")

    # --- mapeamento de alternativas
    def test_palavra_inventada_bloqueia_ok_mas_nao_rejeicao_robusta(self):
        alts = ["Isósceles", "Sesquilátero", "Equilátero", "Escaleno", NDA]
        en = "Triângulo com lados 5, 5 e 8. Qual é o tipo quanto aos lados?"
        self.assertVeredito(_q(en, alts, "A"), "nao_aplicavel")
        self.assertVeredito(_q(en, alts, "D"), "gabarito_errado", sugestao="A")

    def test_lista_de_nomes_nao_mapeia_e_conjuncao_mapeia(self):
        self.assertIsNone(vg._mascara_alternativa("quadrados e retangulos", "quad", "A"))
        self.assertIsNone(vg._mascara_alternativa("retangulo ou losango", "quad", "A"))
        self.assertIsNone(vg._mascara_alternativa("equilatero, isosceles, escaleno", "tri", "A"))
        S, grupo, _ = vg._mascara_alternativa("isosceles e retangulo", "tri", "A")
        self.assertEqual(S, vg.ROTULOS["A"]["isosceles"] & vg.ROTULOS["A"]["retangulo_t"])
        self.assertEqual(grupo, "misto")

    def test_retangulo_entre_nomes_de_figuras_e_ambiguo(self):
        """O protótipo lia "Retângulo" como triângulo retângulo e acusava não unicidade."""
        self.assertVeredito(_q("Uma figura tem três lados e um ângulo reto. Que figura é essa?",
                               ["Quadrado", "Retângulo", "Triângulo", "Losango", NDA], "C"),
                            "nao_aplicavel")

    def test_fora_de_escopo(self):
        self.assertVeredito(_q("A planificação de uma pirâmide de base quadrada tem quantos "
                               "triângulos?", ["1 quadrado e 4 triângulos", "Quadrado",
                                               "Triângulo", "Retângulo", NDA], "A"), "nao_aplicavel")

    # --- condição de existência (9º H16)
    def test_existencia(self):
        self.assertVeredito(_q("É possível formar um triângulo com lados de 2 cm, 3 cm e 6 cm?",
                               ["Sim, escaleno", "Sim, isósceles",
                                "Não é possível formar um triângulo", "Sim, equilátero", NDA], "C"),
                            "ok")
        self.assertVeredito(_q("Existe um triângulo com lados 3 cm, 4 cm e 5 cm. Ele é "
                               "classificado como:", TRI_ANGULOS, "B"), "nao_aplicavel")

    # --- várias figuras no enunciado
    def test_figuras_nomeadas_com_alternativas_compostas(self):
        en = ("O triângulo A tem lados 3, 4 e 5. O triângulo B tem lados 5, 5 e 5. Como se "
              "classificam quanto aos lados?")
        for alts in (["Triângulo A: escaleno; Triângulo B: equilátero",
                      "Triângulo A: isósceles; Triângulo B: equilátero",
                      "Triângulo A: escaleno; Triângulo B: escaleno",
                      "Triângulo A: equilátero; Triângulo B: isósceles", NDA],
                     ["A: escaleno; B: equilátero", "A: isósceles; B: equilátero",
                      "A: escaleno; B: escaleno", "A: equilátero; B: isósceles", NDA]):
            with self.subTest(alts=alts[0]):
                self.assertVeredito(_q(en, alts, "A"), "ok")

    def test_identificadores_falsos_de_entidade(self):
        """"lado A = 8 m" não é entidade (R2-Q1); "comprou 3 bandeiras" também não."""
        self.assertVeredito(_q("Uma barraca triangular tem lado A = 8 m, lado B = 10 m e lado C = "
                               "12 m. Qual é o tipo quanto aos lados?",
                               ["Escaleno", "Isósceles", "Equilátero", "Retângulo", NDA], "A"), "ok")

    def test_contradicao_entre_figuras_nao_e_premissa(self):
        # caso real G-9H17-0036: as frases falam de bandeiras DIFERENTES
        self.assertVeredito(_q("A equipe comprou 3 bandeiras. Duas bandeiras têm 2 lados iguais e "
                               "um lado diferente. A terceira bandeira tem todos os lados iguais. "
                               "Como a equipe classificou a bandeira?",
                               ["Escaleno", "Isósceles", "Equilátero", "Retângulo", NDA], "B"),
                            "nao_aplicavel")

    # --- normalização e leitura
    def test_e_acentuado_nao_vira_conjuncao(self):
        """"equilátero, isósceles e escaleno" não pode virar o fato "é escaleno"."""
        self.assertVeredito(_q("Em um mercado, há três tipos de triângulos: equilátero, isósceles "
                               "e escaleno. Um triângulo tem todos os lados iguais. Como ele é "
                               "chamado?", ["Escaleno", "Retângulo", "Equilátero", "Isósceles",
                                            NDA], "C"), "ok")

    def test_fato_negado_nao_vira_afirmacao(self):
        """O protótipo aplicava "não tem todos os ângulos retos" como "tem"."""
        self.assertVeredito(_q("Um quadrilátero não tem todos os ângulos retos e tem todos os "
                               "lados iguais. Qual é o nome?", QUAD, "B"), "nao_aplicavel")

    def test_dois_lados_opostos_iguais_e_um_par(self):
        det = self.assertVeredito(_q("Um quadrilátero tem dois lados opostos iguais. Qual é o "
                                     "nome?", ["Retângulo", "Losango", "Quadrado", "Trapézio", NDA],
                                     "A"), "dados_insuficientes")
        self.assertIn("1 par de lados opostos iguais", det["fatos"])

    def test_par_de_lados_opostos_paralelos_nao_e_dois_pares(self):
        """O protótipo lia "um par de lados opostos paralelos" também como "dois pares"."""
        v, _ = vg.verificar_geometria(_q("Um quadrilátero tem um par de lados opostos paralelos e "
                                         "dois ângulos retos. Qual é o nome?",
                                         ["Trapézio retângulo", "Retângulo", "Quadrado", "Losango",
                                          NDA], "A"))
        self.assertNotIn(v, REJEITA)

    def test_ordem_ciclica_dos_angulos_depende_de_convencao(self):
        # train_curado (T-9H17-0001): 90, 90, 120, 60 -> trapézio retângulo só se cíclico
        self.assertVeredito(_q("Um quadrilátero tem ângulos internos de 90°, 90°, 120° e 60°. "
                               "Como ele é classificado quanto aos ângulos?",
                               ["Trapézio retângulo", "Retângulo", "Losango", "Quadrado", NDA],
                               "A"), "nao_aplicavel")

    def test_cantos_retos_sao_angulos_retos(self):
        self.assertVeredito(_q("Uma mesa tem a forma de um quadrilátero com todos os cantos "
                               "retos. Qual é o nome?", ["Retângulo", "Losango", "Trapézio",
                                                         "Triângulo", NDA], "A"), "ok")

    def test_unidades_diferentes_sao_convertidas(self):
        self.assertVeredito(_q("Um triângulo tem lados de 50 cm, 0,5 m e 0,5 m. Classifique-o "
                               "quanto aos lados.", TRI_LADOS, "C"), "ok")

    def test_alternativas_copiadas_no_enunciado(self):
        """R2-Q9 traz "A) Escaleno B) Isósceles ..." dentro do enunciado."""
        self.assertVeredito(AUDITADAS["R2-Q9"], "ok")


class TestSemFalsoPositivo(unittest.TestCase):
    """Questões CORRETAS, com fraseado que não estava no corpus de ajuste.

    Cada uma derrubou uma versão anterior do verificador (protótipo ou
    rascunho) com acusação falsa; o comentário diz qual defeito ela trava.
    O veredito pode ser "ok" ou "nao_aplicavel" — nunca reprovação.
    """

    CASOS = (
        # medida decimal arredondada perto da fronteira de Pitágoras (√2, 5√3)
        _q("Um triângulo tem lados de 1 m, 1 m e 1,41 m (aproximadamente). Classifique-o quanto "
           "aos ângulos.", TRI_ANGULOS, "B"),
        _q("O triângulo tem ângulos de 30°, 60° e 90° e lados de 5 cm, 8,66 cm e 10 cm. "
           "Classifique-o quanto aos ângulos.", TRI_ANGULOS, "B"),
        # soma 179,9° por arredondamento
        _q("Um triângulo tem ângulos de 33,3°, 33,3° e 113,3°. Classifique-o quanto aos ângulos.",
           TRI_ANGULOS, "C"),
        # notação que o parser não lê ("=", "∥")
        _q("Em um triângulo, AB = BC = CA. Como ele é classificado?",
           ["Equilátero", "Escaleno", "Retângulo", "Obtusângulo", NDA], "A"),
        _q("Observe o quadrilátero ABCD, em que AB ∥ CD e AD ∦ BC. Qual é o nome?", QUAD, "D"),
        # "iguais dois a dois" é aos pares, não "todos iguais"
        _q("Um quadrilátero tem os lados iguais dois a dois e os quatro ângulos retos. Qual é o "
           "nome?", QUAD, "A"),
        # qualificador que sobra depois do consumo ("de mesma medida")
        _q("Um triângulo tem um ângulo reto e dois ângulos agudos de mesma medida. Classifique-o "
           "quanto aos lados e aos ângulos.", ["Isósceles e retângulo", "Escaleno e retângulo",
                                               "Equilátero e acutângulo",
                                               "Isósceles e obtusângulo", NDA], "A"),
        # perguntas de polaridade invertida
        _q("Um triângulo tem lados de 3 cm, 4 cm e 5 cm. Qual classificação NÃO se aplica a ele?",
           ["Escaleno", "Retângulo", "Equilátero", "Triângulo", NDA], "C"),
        _q("Qual das alternativas apresenta um triângulo impossível?",
           ["Triângulo com dois ângulos retos", "Triângulo com um ângulo reto",
            "Triângulo com três ângulos agudos", "Triângulo com um ângulo obtuso", NDA], "A"),
        _q("Um quadrilátero tem quatro ângulos retos. Qual classificação é incorreta?",
           ["Retângulo", "Paralelogramo", "Trapézio", "Quadrilátero", NDA], "C"),
        _q("Um triângulo tem dois lados iguais. Qual classificação ele NÃO pode ter?",
           ["Isósceles", "Escaleno", "Retângulo", "Acutângulo", NDA], "B"),
        # figura derivada: os fatos do quadrado não são do triângulo
        _q("Um quadrado foi dividido ao meio por uma diagonal. Que tipo de triângulo se forma "
           "quanto aos lados?", TRI_LADOS, "A"),
        # eixo fixado pelo corpo, não pela pergunta
        _q("Os triângulos podem ser equiláteros, isósceles ou escalenos. Um triângulo tem lados "
           "4, 4 e 6. Como ele é classificado?",
           ["Equilátero", "Isósceles", "Escaleno", "Obtusângulo", NDA], "B"),
        # "dois ângulos iguais" num quadrilátero não é modelado (o protótipo cortava tudo)
        _q("Um quadrilátero tem dois ângulos iguais. Qual é o nome?",
           ["Trapézio", "Retângulo", "Quadrado", "Losango", NDA], "E"),
        # referência genérica "de um triângulo" não é segunda figura
        _q("Os ângulos internos de um triângulo somam 180°. Um triângulo tem ângulos de 30°, 60° "
           "e 90°. Classifique-o quanto aos ângulos.", TRI_ANGULOS, "B"),
        # "dois lados medindo 5 cm" não é "todos os lados medem 5 cm"
        _q("Um triângulo tem dois lados medindo 5 cm e o terceiro lado medindo 8 cm. "
           "Classifique-o quanto aos lados.", TRI_LADOS, "A"),
        # lista de ângulos sem "todos" ("os ângulos medem 50°, 60° e 70°"), G-9H17-0126
        _q("Se os ângulos internos do triângulo medem 50°, 60° e 70°, como ele é classificado "
           "quanto aos ângulos?", TRI_ANGULOS, "A"),
        # "9º ano" e "30°C" não são ângulos
        _q("Na turma do 9º ano, a professora desenhou um triângulo com ângulos de 30°, 60° e 90°. "
           "Como ele é classificado quanto aos ângulos?", TRI_ANGULOS, "B"),
        _q("Num dia de 30°C, João desenhou um triângulo com lados 3 cm, 4 cm e 5 cm. Classifique-o "
           "quanto aos ângulos.", TRI_ANGULOS, "B"),
        # rótulo na PERGUNTA é o que se pergunta, não dado
        _q("Em um triângulo, as medidas dos lados são 9 cm, 12 cm e 15 cm. Esse triângulo é "
           "retângulo, acutângulo ou obtusângulo?",
           ["Retângulo", "Acutângulo", "Obtusângulo", "Equilátero", NDA], "A"),
        # medidas de OBJETOS DIFERENTES somadas como se fossem da mesma figura
        _q("A barraca tem formato de triângulo com dois ângulos de 50°. Já a lona da outra "
           "barraca tem um ângulo de 100°. Como se classifica a primeira barraca quanto aos "
           "ângulos?", TRI_ANGULOS, "A"),
        _q("Uma placa triangular tem lados de 3 cm e 4 cm. A outra placa tem um lado de 9 cm. "
           "Classifique a primeira placa quanto aos lados.", TRI_LADOS, "B"),
        _q("Joana desenhou dois triângulos. O primeiro tem um ângulo de 120°. O segundo tem "
           "ângulos de 50° e 60°. Como se classifica o segundo quanto aos ângulos?",
           TRI_ANGULOS, "A"),
        # medida que o parser não leu ("trechos", "da base ... cada") não é "faltam dados"
        _q("Um caminho em forma de triângulo tem trechos de 300 m, 400 m e 500 m. Que tipo de "
           "triângulo o caminho forma quanto aos ângulos?", TRI_ANGULOS, "B"),
        _q("Um triângulo isósceles tem ângulos da base medindo 40° cada. Como ele é classificado "
           "quanto aos ângulos?", TRI_ANGULOS, "C"),
        # identificador romano separa figuras ("figura I ... figura II")
        _q("A figura I tem ângulos de 50° e 60°. A figura II tem um ângulo de 100°. Classifique "
           "a figura I quanto aos ângulos.", TRI_ANGULOS, "A"),
        # "os outros dois ângulos medem 120° e 60°" é lista, não "120° cada"
        _q("Um quadrilátero tem dois ângulos de 90°. Os outros dois ângulos medem 120° e 60°. "
           "Qual é o nome desse quadrilátero?", ["Trapézio", "Retângulo", "Quadrado", "Losango",
                                                 NDA], "A"),
        # várias corretas do banco real em formas diferentes
        _q("Um losango tem um ângulo de 90°. Qual é o nome mais específico desse quadrilátero?",
           QUAD, "C"),
        _q("Um terreno tem a forma de um trapézio com dois ângulos retos. Qual é o tipo de "
           "trapézio?", ["Trapézio isósceles", "Trapézio retângulo", "Trapézio escaleno",
                         "Paralelogramo", NDA], "B"),
        _q("Em um quadrilátero, os lados opostos são paralelos e todos os ângulos são retos. Os "
           "quatro lados têm a mesma medida. Qual é o nome desse quadrilátero?", QUAD, "C"),
    )

    def test_nenhuma_questao_correta_e_reprovada(self):
        for q in self.CASOS:
            with self.subTest(enunciado=q["enunciado"][:70]):
                v, det = vg.verificar_geometria(q)
                self.assertIn(v, ("ok", "nao_aplicavel"), det)


# ---------------------------------------------------------------------------
# Itens reais do banco (data/train_curado.jsonl): nenhum pode ser reprovado.
# ---------------------------------------------------------------------------
TREINO_REAL = {  # data/train_curado.jsonl (banco real), via corpus_treino
    "T-9H17-0002": (
        "banco_real",
        {
            "enunciado": (
                "Observe quadrilátero com lados opostos iguais e todos os ângulos "
                "retos. Como pode ser classificado?"),
            "alternativas": {
                "A": "Losango",
                "B": "Trapézio",
                "C": "Retângulo",
                "D": "Quadrado",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Figura com ângulos retos e lados opostos iguais = retângulo",
            "resposta_correta": "C",
            "difficulty": "MEDIUM",
        }),
    "T-9H17-0003": (
        "banco_real",
        {
            "enunciado": (
                "Como podemos classificar esse quadrilátero? Quadrilátero com dois "
                "pares de lados opostos paralelos e ângulos retos"),
            "alternativas": {
                "A": "Trapézio",
                "B": "Retângulo",
                "C": "Losango",
                "D": "Paralelogramo",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Retângulo tem dois pares de lados paralelos e todos os ângulos retos",
            "resposta_correta": "B",
            "difficulty": "MEDIUM",
        }),
    "T-9H17-0004": (
        "banco_real",
        {
            "enunciado": (
                "Triângulo possui todos os lados iguais. Como ele é classificado "
                "em relação aos lados?"),
            "alternativas": {
                "A": "Escaleno",
                "B": "Isósceles",
                "C": "Equilátero",
                "D": "Retângulo",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Triângulos com todos os lados iguais são equiláteros",
            "resposta_correta": "C",
            "difficulty": "EASY",
        }),
    "T-9H17-0005": (
        "banco_real",
        {
            "enunciado": (
                "Observe quadrado: todos os lados iguais e todos os ângulos retos. "
                "Qual classificação é correta?"),
            "alternativas": {
                "A": "Retângulo",
                "B": "Losango",
                "C": "Quadrado",
                "D": "Trapézio",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Quadrado: todos os lados iguais + todos os ângulos retos",
            "resposta_correta": "C",
            "difficulty": "HARD",
        }),
    "T-9H17-0006": (
        "banco_real",
        {
            "enunciado": (
                "Triângulo possui um ângulo com medida de 90°. Como é classificado "
                "em relação aos ângulos?"),
            "alternativas": {
                "A": "Acutângulo",
                "B": "Obtusângulo",
                "C": "Retângulo",
                "D": "Equilátero",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Triângulo com um ângulo de 90° é classificado como retângulo",
            "resposta_correta": "C",
            "difficulty": "HARD",
        }),
    "T-9H17-0007": (
        "banco_real",
        {
            "enunciado": (
                "Triângulo com lados 5 cm, 5 cm e 8 cm. Como ele é classificado "
                "quanto aos lados?"),
            "alternativas": {
                "A": "Escaleno",
                "B": "Isósceles",
                "C": "Equilátero",
                "D": "Obtusângulo",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Dois lados iguais e um diferente = isósceles",
            "resposta_correta": "B",
            "difficulty": "MEDIUM",
        }),
    "T-9H17-0008": (
        "banco_real",
        {
            "enunciado": (
                "Como o triângulo de Luana pode ser classificado? Triângulo com "
                "dois lados iguais e um ângulo de 90°"),
            "alternativas": {
                "A": "Isósceles e acutângulo",
                "B": "Escaleno e retângulo",
                "C": "Isósceles e retângulo",
                "D": "Equilátero e retângulo",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Dois lados iguais (isósceles) e um ângulo de 90° (retângulo)",
            "resposta_correta": "C",
            "difficulty": "HARD",
        }),
    "T-9H17-0001": (
        "destilado",
        {
            "enunciado": (
                "Na aula de culinária, Ana usou uma forma de bolo com base em "
                "formato de quadrilátero. Ela mediu os ângulos internos e "
                "encontrou: 90°, 90°, 120° e 60°. Como essa forma deve ser "
                "classificada quanto aos ângulos internos?"),
            "alternativas": {
                "A": "Trapézio retângulo",
                "B": "Retângulo",
                "C": "Losango",
                "D": "Quadrado",
                "E": "Paralelogramo qualquer",
            },
            "resolucao_passo_a_passo": (
                "A soma dos ângulos internos de um quadrilátero é 360° "
                "(90+90+120+60 = 360). Como há exatamente dois ângulos retos e os "
                "outros dois são diferentes entre si (120° e 60°), trata-se de um "
                "trapézio que possui dois ângulos retos, chamado trapézio "
                "retângulo. Retângulo e quadrado exigiriam quatro ângulos retos; "
                "losango e paralelogramo qualquer não exigem ângulo reto."),
            "resposta_correta": "A",
            "difficulty": "HARD",
        }),
    "T-9H16-0014": (
        "banco_real",
        {
            "enunciado": (
                "Triângulo com ângulo de 90° e lados 5 cm, 12 cm e 13 cm. Esse "
                "triângulo é:"),
            "alternativas": {
                "A": "Equilátero",
                "B": "Isósceles",
                "C": "Retângulo",
                "D": "Obtusângulo",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Obedece Pitágoras: 5²+12²=25+144=169=13² → triângulo retângulo",
            "resposta_correta": "C",
            "difficulty": "HARD",
        }),
    "T-9H16-0016": (
        "banco_real",
        {
            "enunciado": (
                "Triângulo com lados 6 cm, 6 cm e 8 cm. Como ele é classificado em "
                "relação aos lados?"),
            "alternativas": {
                "A": "Escaleno",
                "B": "Equilátero",
                "C": "Retângulo",
                "D": "Isósceles",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "Dois lados iguais e um diferente = triângulo isósceles",
            "resposta_correta": "D",
            "difficulty": "MEDIUM",
        }),
    "T-9H16-0013": (
        "banco_real",
        {
            "enunciado": (
                "É possível construir esse triângulo com essas medidas? Triângulo "
                "com lados 4 cm, 7 cm e 12 cm"),
            "alternativas": {
                "A": "Sim, pois todos os lados são diferentes",
                "B": "Sim, pois a soma dos lados é maior que 12 cm",
                "C": "Não, pois a soma de dois lados não é maior que o terceiro",
                "D": "Não, pois triângulos devem ter lados iguais",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": "A soma de 4 + 7 = 11 < 12, portanto não forma triângulo",
            "resposta_correta": "C",
            "difficulty": "MEDIUM",
        }),
    "T-9H16-0012": (
        "banco_real",
        {
            "enunciado": (
                "Pedro desenhou triângulo com lados 5 cm, 7 cm e 13 cm. Esse "
                "triângulo pode ser construído?"),
            "alternativas": {
                "A": "Sim, todos lados diferentes",
                "B": "Sim, soma>13",
                "C": "Não, soma de 2 lados ≤ terceiro",
                "D": "Não, não múltiplos de 3",
                "E": "Nenhuma das alternativas anteriores",
            },
            "resolucao_passo_a_passo": (
                "Condição de existência: soma de 2 lados > terceiro. Aqui: "
                "5+7=12<13, logo não pode"),
            "resposta_correta": "C",
            "difficulty": "MEDIUM",
        }),
}


class TestTreinoReal(unittest.TestCase):
    def test_banco_real_nunca_e_reprovado(self):
        for nome, (_fonte, q) in sorted(TREINO_REAL.items()):
            with self.subTest(nome):
                v, det = vg.verificar_geometria(q)
                self.assertNotIn(v, REJEITA, det)

    def test_classificacoes_verificaveis_do_banco_dao_ok(self):
        verificaveis = ("T-9H17-0002", "T-9H17-0004", "T-9H17-0005", "T-9H17-0006",
                        "T-9H17-0007", "T-9H17-0008", "T-9H16-0014", "T-9H16-0016")
        for nome in verificaveis:
            with self.subTest(nome):
                v, det = vg.verificar_geometria(TREINO_REAL[nome][1])
                self.assertEqual(v, "ok", det)


@unittest.skipUnless((ROOT / "data" / "train_curado.jsonl").exists(), "data/ ausente")
class TestDadosDeTreino(unittest.TestCase):
    """Zero reprovação em data/*.jsonl (só leitura). É a medida de falso
    positivo em escala: ~9 mil questões de todas as habilidades."""

    ARQUIVOS = ("train_curado.jsonl", "train.jsonl", "train_multi.jsonl", "distill.jsonl",
                "val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl")

    def test_nenhuma_reprovacao(self):
        import schema_utils as su
        lidas = 0
        for nome in self.ARQUIVOS:
            caminho = ROOT / "data" / nome
            if not caminho.exists():
                continue
            with self.subTest(arquivo=nome), caminho.open(encoding="utf-8") as fh:
                reprovadas = []
                for n, linha in enumerate(fh, 1):
                    try:
                        msgs = json.loads(linha)["messages"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    obj = su.parse_json(msgs[-1]["content"]) if msgs else None
                    for q in su.extract_questoes(obj):
                        lidas += 1
                        v, det = vg.verificar_geometria(q)
                        if v in REJEITA:
                            reprovadas.append((n, v, det.get("motivo")))
                self.assertEqual(reprovadas, [])
        # guarda contra passar no vazio (formato do arquivo mudou e nada foi lido)
        self.assertGreater(lidas, 1000)


class TestContrato(unittest.TestCase):
    def test_entradas_malformadas_nunca_levantam(self):
        ruins = (None, {}, {"alternativas": None}, {"alternativas": {"A": "x"}},
                 {"alternativas": {"A": None, "B": 3}, "resposta_correta": "A"},
                 {"enunciado": None, "alternativas": {"A": "Losango", "B": "Quadrado"},
                  "resposta_correta": "B"},
                 _q("", ["", "", "", "", ""], "A"), "texto")
        for q in ruins:
            with self.subTest(q=repr(q)[:40]):
                v, det = vg.verificar_geometria(q)
                self.assertIn(v, vg.VEREDITOS)
                self.assertNotEqual(det.get("motivo"), "erro_interno", det)

    def test_vereditos_e_detalhe(self):
        self.assertLessEqual(REJEITA, set(vg.VEREDITOS))
        self.assertNotIn("ok", REJEITA)
        self.assertEqual(vg.MODO_GEOMETRIA, "sombra")  # integração começa em sombra
        for nome, q in AUDITADAS.items():
            with self.subTest(nome):
                v, det = vg.verificar_geometria(q)
                self.assertIn("motivo", det)
                self.assertIn("explicacao", det)
                self.assertNotEqual(det["motivo"], "erro_interno")

    def test_deterministico(self):
        for nome, q in AUDITADAS.items():
            with self.subTest(nome):
                self.assertEqual(vg.verificar_geometria(q), vg.verificar_geometria(q))

    def test_invariante_a_permutacao_das_alternativas(self):
        """A letra viaja com o texto: veredito igual e sugestão aponta o mesmo texto.
        Usa a permutação de PRODUÇÃO (gerar_lote.permutar_alternativas)."""
        import gerar_lote as gl
        for nome, q in AUDITADAS.items():
            v0, d0 = vg.verificar_geometria(q)
            for seed in (1, 7, 42, 2026):
                with self.subTest(nome, seed=seed):
                    p = gl.permutar_alternativas(q, seed)
                    v1, d1 = vg.verificar_geometria(p)
                    self.assertEqual(v1, v0)
                    if d0.get("sugestao"):
                        self.assertEqual(p["alternativas"][d1["sugestao"]],
                                         q["alternativas"][d0["sugestao"]])

    def test_fixtures_de_geometria_existentes_nao_mudam_de_sentido(self):
        """Fixtures de outros testes que encostam em geometria: o "sem_conta"
        de test_consistencia_multipasso (alternativas "a".."e") fica fora do
        verificador; Q_TRI/Q_QUAD de test_distill são corretas (ok)."""
        sem_conta = {"enunciado": "x", "alternativas": dict(zip("ABCDE", "abcde")),
                     "resolucao_passo_a_passo": "É um triângulo obtusângulo.",
                     "resposta_correta": "A", "difficulty": "EASY"}
        self.assertEqual(vg.verificar_geometria(sem_conta)[0], "nao_aplicavel")
        alts_tri = ["Isósceles", "Escaleno", "Equilátero", "Retângulo", NDA]
        q_tri = _q("Um triângulo tem lados de 5 cm, 5 cm e 7 cm. Como ele é classificado quanto "
                   "aos lados?", alts_tri, "A")
        q_quad = _q("Um quadrilátero tem quatro lados iguais e quatro ângulos retos. Que "
                    "quadrilátero é esse?", ["Quadrado", "Trapézio", "Losango", "Paralelogramo",
                                             NDA], "A")
        self.assertEqual(vg.verificar_geometria(q_tri)[0], "ok")
        self.assertEqual(vg.verificar_geometria(q_quad)[0], "ok")


class TestPortabilidade(unittest.TestCase):
    """Tudo que roda na inferência vira TypeScript no app: nada de lookbehind,
    grupo nomeado, flag embutida ou classe Unicode; \\b e \\w com semântica ASCII."""

    def test_regex_sem_construcoes_nao_portaveis(self):
        fonte = (ROOT / "src" / "verificador_geometria.py").read_text(encoding="utf-8")
        codigo = "\n".join(l for l in fonte.splitlines() if not l.lstrip().startswith("#"))
        for proibido in ("(?<=", "(?<!", "(?P<", "(?i)", "(?s)", "(?m)", "(?x)", "\\p{", "(?>"):
            with self.subTest(proibido=proibido):
                self.assertNotIn(proibido, codigo)

    def test_regex_com_palavra_usam_ascii(self):
        for nome, valor in vars(vg).items():
            if isinstance(valor, re.Pattern) and re.search(r"\\[bwd]", valor.pattern):
                with self.subTest(nome):
                    self.assertTrue(valor.flags & re.ASCII, nome)

    # Revisão adversarial de 2026-10-01: o teste acima só olhava "(?i)" literal
    # e só este módulo. (?i:...) é ES2025 (Hermes/JSC antigos dão SyntaxError),
    # "{,n}" é literal no JS e {0,n} no Python, \Z não existe no JS, e as
    # construções proibidas moravam em gerar_lote/diversidade (caminho de
    # inferência), não aqui.
    PROIBIDOS = ("(?<=", "(?<!", "(?P<", "(?i", "(?a", "(?u", "(?L", "(?s", "(?m", "(?x",
                 "\\p{", "(?>", "{,", "\\Z")
    MODULOS_DE_INFERENCIA = ("src/verificador_geometria.py", "src/gerar_lote.py",
                             "src/diversidade.py", "tests/test_model.py")

    def test_modulos_de_inferencia_sem_construcoes_nao_portaveis(self):
        for rel in self.MODULOS_DE_INFERENCIA:
            fonte = (ROOT / rel).read_text(encoding="utf-8")
            codigo = "\n".join(l for l in fonte.splitlines() if not l.lstrip().startswith("#"))
            for proibido in self.PROIBIDOS:
                with self.subTest(rel, proibido=proibido):
                    self.assertNotIn(proibido, codigo)

    def test_palavras_chave_da_taxonomia_sem_construcoes_nao_portaveis(self):
        # palavras_chave e aplica_a viajam no JSON e são casadas no app.
        import build_taxonomia as bt
        tax = bt.construir()
        pats = [(k, s["id"], p) for k, h in tax["habilidades"].items() for s in h["subtemas"]
                for p in (s["palavras_chave"] if isinstance(s["palavras_chave"], list)
                          else [s["palavras_chave"]])]
        pats += [("tipos", t["id"], t.get("aplica_a") or "") for t in tax.get("tipos_raciocinio", [])]
        self.assertGreater(len(pats), 100)
        for k, sid, p in pats:
            for proibido in self.PROIBIDOS:
                self.assertNotIn(proibido, p or "", (k, sid))

    def test_padroes_executados_sao_portaveis(self):
        """Instrumenta re._compile durante a verificação: pega também os padrões
        montados na hora (eixo, entidades) e os de tuplas (_VOCAB_RE)."""
        import re as _re
        vistos = {}
        original = _re._compile

        def espiao(pattern, flags):
            if isinstance(pattern, str):
                vistos[(pattern, int(flags))] = True
            return original(pattern, flags)

        questoes = list(AUDITADAS.values()) + list(TestSemFalsoPositivo.CASOS)
        from unittest import mock
        with mock.patch.object(_re, "_compile", espiao):
            for q in questoes:
                vg.verificar_geometria(q)
        padroes = set(vistos)
        for valor in vars(vg).values():
            if isinstance(valor, _re.Pattern):
                padroes.add((valor.pattern, valor.flags))
        for p, _r in vg._VOCAB_RE + vg._REESCRITAS:
            padroes.add((p.pattern, p.flags))
        self.assertGreater(len(padroes), 100)
        for p, flags in padroes:
            with self.subTest(p=p[:60]):
                for proibido in self.PROIBIDOS:
                    self.assertNotIn(proibido, p)
                if _re.search(r"\\[bwd]", p):
                    self.assertTrue(flags & _re.ASCII, p)

    def test_texto_normalizado_e_ascii_nas_palavras(self):
        t = vg._minusculas(vg._normalizar("Triângulo é isósceles; ÂNGULOS de 90º e lado ≥ 2"))
        self.assertEqual(t, "triangulo eh isosceles; angulos de 90° e lado ≥ 2")


class TestCorrecoesRevisaoAdversarial(unittest.TestCase):
    """Defeitos CONFIRMADOS pela revisão adversarial de 2026-10-01 (sondas
    escritas fora de qualquer corpus). Cada teste falhava antes da correção;
    as questões "corretas" daqui nunca podem ser reprovadas e as erradas
    continuam reprovadas (a correção é pela regra geral, não por exceção)."""

    def assertNaoReprova(self, q):
        v, det = vg.verificar_geometria(q)
        self.assertNotIn(v, REJEITA, det)
        return v, det

    def assertReprova(self, q, esperado=None):
        v, det = vg.verificar_geometria(q)
        self.assertIn(v, REJEITA, det)
        if esperado:
            self.assertEqual(v, esperado, det)
        return det

    # --- "os lados iguais" com artigo definido e SEM quantificador é o PAR do
    # isósceles, não "todos os lados iguais" (5 sondas corretas reprovadas,
    # uma com sugestão de letra errada).
    def test_lados_iguais_atributivo_nao_e_equilatero(self):
        casos = (
            _q("Em um triângulo, os lados iguais medem 5 cm e o outro lado mede 8 cm. Quanto aos "
               "ângulos, ele é:", TRI_ANGULOS, "C"),
            _q("Um triângulo tem lados iguais de 6 cm e um lado de 4 cm. Quanto aos lados, ele é:",
               TRI_LADOS, "A"),
            _q("Um triângulo isósceles tem o ângulo entre os lados iguais medindo 100°. Quanto aos "
               "ângulos, ele é:", TRI_ANGULOS, "C"),
            _q("Um triângulo isósceles tem os lados iguais medindo 5 cm e a base 8 cm. Quanto aos "
               "ângulos, ele é:", TRI_ANGULOS, "C"),
            _q("Num telhado triangular, os lados iguais têm 4 m e o terceiro lado tem 6 m. "
               "Classifique-o quanto aos lados.", TRI_LADOS, "A"),
            _q("Uma placa tem forma de triângulo isósceles cujos ângulos iguais medem 50° cada. "
               "Quanto aos ângulos, ela é:", TRI_ANGULOS, "A"),
        )
        for q in casos:
            with self.subTest(q["enunciado"][:60]):
                self.assertNaoReprova(q)

    def test_angulos_iguais_atributivo_nao_sugere_letra_errada(self):
        # 30° + 30° + 120°: obtusângulo é o CERTO; antes saía gabarito_errado
        # com sugestão "Acutângulo".
        v, det = self.assertNaoReprova(_q("Em um triângulo, os ângulos iguais medem 30° cada um. "
                                          "Quanto aos ângulos, ele é:", TRI_ANGULOS, "C"))
        self.assertIsNone(det.get("sugestao"))

    def test_lados_iguais_predicativo_continua_todos(self):
        # "são iguais", "todos", "tem os lados iguais." continuam universais.
        self.assertReprova(_q("Em um triângulo, os lados são iguais. Quanto aos lados, ele é:",
                              TRI_LADOS, "B"), "gabarito_errado")
        self.assertReprova(_q("Um quadrilátero tem os lados iguais e um ângulo de 60°. Qual é o "
                              "nome mais específico?", QUAD, "C"))
        self.assertReprova(_q("Um triângulo tem os ângulos iguais. Quanto aos ângulos, ele é:",
                              TRI_ANGULOS, "B"), "gabarito_errado")

    # --- nome de figura do CONTEXTO ("pátio em forma de quadrado") não é fato
    # da figura perguntada quando outro objeto aparece depois dele (7/7
    # questões corretas reprovadas).
    def test_rotulo_de_outro_objeto_nao_e_fato(self):
        alts = ["Quadrado", "Losango", "Retângulo", "Trapézio", NDA]
        casos = (
            _q("Uma escola tem um pátio em forma de quadrado. No pátio, os alunos pintaram um "
               "quadrilátero com quatro ângulos retos e lados de 3 m e 6 m. Qual é o nome do "
               "quadrilátero pintado?", alts, "C"),
            _q("Joana tem um caderno em formato de retângulo. Nele, desenhou um quadrilátero com "
               "os quatro lados iguais e sem ângulos retos. Qual é o nome desse quadrilátero?",
               ["Quadrado", "Retângulo", "Losango", "Trapézio", NDA], "C"),
            _q("Uma escola tem um pátio em formato de trapézio. No pátio foi pintado um "
               "quadrilátero com quatro ângulos retos e lados 2 m e 5 m. Qual é o nome do "
               "quadrilátero pintado?", alts, "C"),
            _q("Numa feira, uma barraca tem lona em formato de losango. Ao lado, há uma mesa com "
               "quatro ângulos retos e lados de 1 m e 2 m. Qual é a forma da mesa?", alts, "C"),
            _q("Um terreno em forma de trapézio foi dividido. Uma das partes é um quadrilátero "
               "com quatro lados iguais e quatro ângulos retos. Qual é o nome dessa parte?",
               ["Retângulo", "Losango", "Quadrado", "Trapézio", NDA], "C"),
            _q("Uma toalha em formato de quadrado cobre uma mesa. Sobre ela há um guardanapo com "
               "quatro lados iguais e ângulos de 60° e 120°. Qual é a forma do guardanapo?",
               alts, "B"),
        )
        for q in casos:
            with self.subTest(q["enunciado"][:60]):
                v, det = self.assertNaoReprova(q)
                self.assertFalse(any(f.startswith("rótulo") for f in det.get("fatos", [])), det)

    def test_rotulo_da_propria_figura_continua_fato(self):
        alts = ["Losango", "Trapézio", "Quadrado", "Triângulo", NDA]
        det = vg.verificar_geometria(_q("Uma tampa tem formato de quadrado. Qual é o nome dessa "
                                        "figura?", alts, "C"))[1]
        self.assertEqual(det["motivo"], "gabarito_garantido_e_mais_especifico", det)
        self.assertReprova(_q("Uma mesa em formato de quadrado tem lados de 3 cm, 4 cm, 3 cm e "
                              "4 cm. Qual é o nome dela?", alts, "C"), "premissa_impossivel")

    def test_contexto_nao_vira_motivo_nao_unica(self):
        # Antes: nao_unica "garantidas: A, D" porque o "retângulo" do contexto
        # entrava como fato. O motivo enganoso não pode mais sair.
        v, det = vg.verificar_geometria(_q(
            "Um comerciante vendeu produtos em formato de retângulo. Qual dos seguintes é um "
            "exemplo de um quadrilátero com todos os lados iguais?",
            ["Retângulo", "Trapézio", "Triângulo", "Losango", NDA], "A"))
        self.assertNotEqual(v, "nao_unica", det)
        self.assertNotEqual(v, "ok", det)

    # --- "Qual dos seguintes..." é a PERGUNTA, não sinal de várias figuras.
    def test_seguintes_na_pergunta_nao_abafa_premissa_impossivel(self):
        for perg in ("Qual dos seguintes tipos de quadrilátero representa o jardim?",
                     "Qual das seguintes alternativas classifica o jardim?"):
            with self.subTest(perg):
                self.assertReprova(_q("Um jardim tem 4 lados iguais, um ângulo obtuso e 4 ângulos "
                                      "retos. " + perg, QUAD, "C"), "premissa_impossivel")

    # --- condição de existência: gabarito que NEGA a existência é correto.
    def test_existencia_negada_no_gabarito_nao_reprova(self):
        casos = (
            _q("Um triângulo tem dois ângulos retos. Qual é a classificação desse triângulo?",
               ["Retângulo", "Obtusângulo", "Acutângulo", "Equilátero",
                "Esse triângulo não existe"], "E"),
            _q("Pedro desenhou um triângulo com lados 5 cm, 7 cm e 13 cm. Como esse triângulo é "
               "classificado quanto aos lados?", ["Equilátero", "Isósceles", "Escaleno",
                                                  "Esses lados não formam um triângulo", NDA], "D"),
            _q("Ana disse que os ângulos de um triângulo medem 90°, 60° e 40°. O que se pode "
               "afirmar?", ["É um triângulo retângulo", "É um triângulo acutângulo",
                            "É um triângulo obtusângulo",
                            "Esse triângulo não existe, pois a soma dos ângulos é 190°",
                            "É um triângulo isósceles"], "D"),
            _q("Um quadrilátero tem quatro ângulos obtusos. Qual é o nome desse quadrilátero?",
               ["Quadrado", "Losango", "Trapézio", "Não existe quadrilátero assim", NDA], "D"),
            # sim/não e NDA: abstém (não há classificação a conferir)
            _q("Um triângulo pode ter dois ângulos obtusos?",
               ["Sim, é obtusângulo", "Sim, é acutângulo",
                "Não, pois a soma seria maior que 180°", "Sim, é retângulo", NDA], "C"),
            _q("Qual triângulo possui dois ângulos obtusos?",
               ["Obtusângulo", "Acutângulo", "Retângulo", "Equilátero", NDA], "E"),
        )
        for q in casos:
            with self.subTest(q["enunciado"][:60]):
                self.assertNaoReprova(q)
        v, det = vg.verificar_geometria(casos[0])
        self.assertEqual((v, det["motivo"]), ("ok", "existencia_negada_corretamente"))

    def test_premissa_impossivel_com_gabarito_classificacao_continua(self):
        self.assertReprova(_q("Um triângulo tem dois ângulos retos. Qual é a classificação desse "
                              "triângulo?", ["Retângulo", "Obtusângulo", "Acutângulo",
                                             "Equilátero", "Esse triângulo não existe"], "A"),
                           "premissa_impossivel")

    # --- medida por extenso não é "falta de dado".
    def test_medida_por_extenso_nao_e_dados_insuficientes(self):
        casos = (
            _q("Um triângulo tem um ângulo de noventa graus. Quanto aos ângulos, ele é:",
               TRI_ANGULOS, "B"),
            _q("Um triângulo tem um ângulo de cento e vinte graus. Quanto aos ângulos, ele é:",
               TRI_ANGULOS, "C"),
            _q("Um triângulo tem lados de cinco, cinco e cinco centímetros. Quanto aos lados, "
               "ele é:", TRI_LADOS, "C"),
            _q("Os ângulos de uma placa triangular medem quarenta graus, cinquenta graus e "
               "noventa graus. Quanto aos ângulos, ela é:", TRI_ANGULOS, "B"),
        )
        for q in casos:
            with self.subTest(q["enunciado"][:60]):
                self.assertNaoReprova(q)

    def test_digitos_de_largura_total_sao_lidos(self):
        det = vg.verificar_geometria(_q("Um triângulo tem lados de ３ cm, ４ cm e ５ cm. Quanto aos "
                                        "ângulos, ele é:", TRI_ANGULOS, "B"))[1]
        self.assertEqual(det["motivo"], "gabarito_garantido_e_mais_especifico", det)

    # --- separador de milhar pt-BR ("1.000" é mil, não 1,000).
    def test_separador_de_milhar(self):
        self.assertEqual(vg.verificar_geometria(_q(
            "Um terreno triangular tem lados de 1.000 m, 800 m e 600 m. Quanto aos ângulos, esse "
            "triângulo é:", TRI_ANGULOS, "B"))[0], "ok")
        self.assertEqual(vg.verificar_geometria(_q(
            "Um terreno triangular tem lados de 1.200 m, 1.200 m e 900 m. Quanto aos lados, esse "
            "triângulo é:", TRI_LADOS, "A"))[0], "ok")
        # "1 000" (espaço de milhar) é ambíguo com lista: abstém.
        self.assertEqual(vg.verificar_geometria(_q(
            "Um terreno triangular tem lados de 1 000 m, 800 m e 600 m. Quanto aos lados, esse "
            "triângulo é:", TRI_LADOS, "B"))[0], "nao_aplicavel")
        # decimal com vírgula continua decimal
        self.assertEqual(vg.verificar_geometria(_q(
            "Um triângulo tem lados de 1,5 m, 2 m e 2,5 m. Quanto aos ângulos, ele é:",
            TRI_ANGULOS, "B"))[0], "ok")

    # --- terna pitagórica decimal EXATA não é arredondamento (lacuna de recall).
    def test_terna_pitagorica_decimal_exata(self):
        for lados in ("0,3 m, 0,4 m e 0,5 m", "2,5 cm, 6 cm e 6,5 cm"):
            with self.subTest(lados):
                self.assertEqual(vg.verificar_geometria(_q(
                    f"Os lados de um triângulo medem {lados}. Quanto aos ângulos, ele é:",
                    TRI_ANGULOS, "B"))[0], "ok")
        # o arredondamento perto da fronteira continua sem veredito
        self.assertEqual(vg.verificar_geometria(_q(
            "Os lados de um triângulo medem 1 m, 1 m e 1,41 m. Quanto aos ângulos, ele é:",
            TRI_ANGULOS, "A"))[0], "nao_aplicavel")

    # --- teto de magnitude: o TypeScript usa Number (exato só até 2^53).
    def test_medida_gigante_abstem(self):
        self.assertEqual(vg.verificar_geometria(_q(
            "Um triângulo tem lados de 6074549952 mm, 8101350014 mm e 10125810050 mm. Quanto aos "
            "ângulos, ele é:", TRI_ANGULOS, "B"))[0], "nao_aplicavel")

    # --- caixa alta: "A" artigo e "A" entidade se confundem.
    def test_caixa_alta_nao_reprova(self):
        q = AUDITADAS["R1-Q1"]
        alto = dict(q, enunciado=q["enunciado"].upper(),
                    alternativas={k: v.upper() for k, v in q["alternativas"].items()},
                    resolucao_passo_a_passo=q["resolucao_passo_a_passo"].upper())
        self.assertEqual(vg.verificar_geometria(alto)[0], "nao_aplicavel")

    # --- corte de sufixo sem regex quadrática (Hermes não tem JIT de regex).
    def test_corte_de_sufixo_linear_e_equivalente(self):
        import time
        antigo = re.compile(r"(?:[\s,;]+(?:e\s+)?(?:a|o|as|os))?[\s,;.]*$", re.ASCII)
        for t in ("um triangulo agudo, a", "escaleno e o", "retangulo", "obtusangulo.",
                  "isosceles ,; e os", "triangulo retangulo e a", "quadrado ;", "losango e",
                  "agudo,a", "x  e  as  ", ""):
            with self.subTest(t=t):
                self.assertEqual(vg._cortar_sufixo(t), antigo.sub("", t, count=1))
        t0 = time.perf_counter()
        vg._cortar_sufixo("x" + " " * 20000 + "y")
        self.assertLess(time.perf_counter() - t0, 0.05)


if __name__ == "__main__":
    unittest.main()


class TestD5DoUsuario(unittest.TestCase):
    """verificar_d5 (P1, 2026-10-01): a D5 do usuário — só o gabarito pode ser
    verdadeiro, por qualquer critério — sobre o detalhe de verificar_geometria.
    É o pré-filtro da injeção ANTES dos juízes pagos. Casos reais: candidatas do
    9º H17 dos pilotos (outputs/injecao_rejeitadas.jsonl), reescritas aqui
    porque outputs/ é gitignored."""

    def _d5(self, q):
        return vg.verificar_d5(q)

    def test_outro_eixo_acutangulo_quanto_aos_lados(self):
        # o exemplo do usuário: 6-6-7 "quanto aos lados" com "Acutângulo"
        q = _q("Um triângulo tem lados de 6 cm, 6 cm e 7 cm. Quanto aos lados, ele é:",
               ["Equilátero", "Isósceles", "Escaleno", "Acutângulo", "Obtusângulo"], "B")
        self.assertEqual(vg.verificar_geometria(q)[0], "ok")  # a convenção 4/5 aceita
        viola, det = self._d5(q)
        self.assertTrue(viola)
        self.assertEqual((det["outro_eixo"], det["superclasse"]), (["D"], []))

    def test_superclasse_isosceles_no_equilatero(self):
        q = _q("Um triângulo tem os três lados iguais. Como se chama esse triângulo?",
               ["Escaleno", "Isósceles", "Retângulo", "Equilátero", "Obtusângulo"], "D")
        viola, det = self._d5(q)
        self.assertTrue(viola)
        self.assertEqual(det["superclasse"], ["B"])

    def test_superclasse_paralelogramo_no_losango_mesmo_com_abstencao(self):
        # INJ-9-H17-F-00011 (piloto 1): o verificador se abstém (quadrado/retângulo
        # ficam I), mas "Paralelogramo" é V de forma robusta
        q = _q("Na biblioteca da escola, um cartaz tem formato de quadrilátero com quatro lados de mesma "
               "medida e dois pares de lados paralelos. Não há informação sobre ângulos retos. Qual é a "
               "classificação mais específica garantida por esses dados?",
               ["quadrado", "losango", "retângulo", "trapézio", "paralelogramo"], "B")
        self.assertEqual(vg.verificar_geometria(q)[0], "nao_aplicavel")
        viola, det = self._d5(q)
        self.assertTrue(viola)
        self.assertEqual(det["superclasse"], ["E"])

    def test_escaleno_verdadeiro_numa_pergunta_quanto_aos_angulos(self):
        # INJ-9-H17-F-00005: 30-60-90 "quanto aos ângulos" com "Escaleno"
        q = _q("Na barraca de legumes da feira, o vendedor usou uma tábua triangular para apoiar caixas. "
               "Os ângulos dessa tábua medem 30°, 60° e 90°. Como esse triângulo se classifica quanto aos "
               "ângulos?", ["Retângulo", "Acutângulo", "Obtusângulo", "Equilátero", "Escaleno"], "A")
        viola, det = self._d5(q)
        self.assertTrue(viola)
        self.assertEqual(det["outro_eixo"], ["E"])

    def test_controles_com_todos_os_distratores_falsos(self):
        for q in (
            _q("Um triângulo tem lados de 5 cm, 5 cm e 6 cm. Quanto aos lados, ele é:",
               ["Equilátero", "Isósceles", "Escaleno", "Retângulo", "Obtusângulo"], "B"),
            _q("Em uma receita de bolo, uma fatia triangular foi cortada de modo que seus três ângulos "
               "internos medem 50°, 60° e 70°. Como esse triângulo deve ser classificado quanto aos ângulos?",
               ["Retângulo", "Obtusângulo", "Acutângulo", "Equilátero", "Isósceles"], "C"),
        ):
            with self.subTest(q["enunciado"][:40]):
                self.assertEqual(self._d5(q), (False, None))

    def test_nao_se_pronuncia_quando_o_veredito_principal_ja_reprova_ou_se_abstem_por_outro_motivo(self):
        # premissa impossível: quem reprova é verificar_geometria
        q = _q("Um triângulo tem dois ângulos retos. Quanto aos lados, ele é:",
               ["Equilátero", "Isósceles", "Escaleno", "Retângulo", "Obtusângulo"], "B")
        self.assertIn(vg.verificar_geometria(q)[0], REJEITA)
        self.assertEqual(self._d5(q), (False, None))
        # fora de escopo / sem rótulo de figura
        self.assertEqual(self._d5(_q("Quanto é 2 + 3?", ["5", "6", "7", "8", "9"], "A")), (False, None))
        # resultado já calculado é reaproveitado (sem refazer o parse)
        q2 = _q("Um triângulo tem os três lados iguais. Como se chama esse triângulo?",
                ["Escaleno", "Isósceles", "Retângulo", "Equilátero", "Obtusângulo"], "D")
        self.assertTrue(vg.verificar_d5(q2, vg.verificar_geometria(q2))[0])

    def test_nao_muda_o_veredito_publico(self):
        # verificar_d5 só LÊ o detalhe; verificar_geometria continua na convenção 4
        q = _q("Um triângulo tem os três lados iguais. Como se chama esse triângulo?",
               ["Escaleno", "Isósceles", "Retângulo", "Equilátero", "Obtusângulo"], "D")
        self.assertEqual(vg.verificar_geometria(q)[0], "ok")
