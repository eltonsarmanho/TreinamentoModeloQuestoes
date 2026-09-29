"""
Testes do mecanismo de diversidade (src/diversidade.py) e da taxonomia gerada.
Compatível com unittest e pytest:
    venv/bin/python -m unittest tests/test_diversidade.py -v
"""
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_taxonomia  # noqa: E402
import diversidade as dv  # noqa: E402

TAX = build_taxonomia.construir()  # sempre a partir do banco (não depende do JSON em disco)


def K(ano, hab):
    return len(TAX["habilidades"][f"{ano}|{hab}"]["subtemas"])


def plano(ano, hab, n, seed=0, **kw):
    return dv.planejar_lote(ano, hab, n, "Moderado", seed, taxonomia=TAX, **kw)


def q(enunciado, alts=("a", "b", "c", "d", "Nenhuma das alternativas anteriores")):
    return {"enunciado": enunciado, "alternativas": dict(zip("ABCDE", alts)),
            "resolucao_passo_a_passo": "", "resposta_correta": "A", "difficulty": "MEDIUM"}


class TestTaxonomia(unittest.TestCase):
    def test_todas_as_habilidades_de_matematica(self):
        import sqlite3
        con = sqlite3.connect(ROOT / "DB" / "questoes.db")
        pares = {f"{a}|{h}" for a, h in con.execute(
            "SELECT DISTINCT ano, habilidade FROM itens WHERE disciplina='Matemática'")}
        self.assertEqual(pares, set(TAX["habilidades"]))
        for h in TAX["habilidades"].values():
            self.assertGreaterEqual(len(h["subtemas"]), 1)
            for s in h["subtemas"]:
                for campo in ("id", "rotulo", "palavras_chave", "tipos_raciocinio", "n_exemplos_db"):
                    self.assertIn(campo, s)

    def test_contextos_sao_globais_e_nao_subtema(self):
        ctx = {c["id"] for c in TAX["contextos"]}
        for h in TAX["habilidades"].values():
            self.assertFalse(ctx & {s["id"] for s in h["subtemas"]})

    def test_mesmo_codigo_significa_coisas_diferentes_por_ano(self):
        s9 = {s["id"] for s in TAX["habilidades"]["9º|H17"]["subtemas"]}
        s5 = {s["id"] for s in TAX["habilidades"]["5º|H17"]["subtemas"]}
        self.assertEqual(s9, {"triangulo", "quadrilatero"})
        self.assertEqual(s5, {"relogio_analogico", "relogio_digital"})
        self.assertFalse(s9 & s5)

    def test_subtemas_geometria_9(self):
        esperado = {
            "H16": {"condicao_existencia", "soma_angulos_internos", "angulo_externo"},
            "H18": {"raio_diametro", "corda", "arco", "angulo_central_inscrito"},
            "H20": {"retangulo_quadrado", "triangulo", "circulo"},
            "H21": {"retangulo_quadrado", "triangulo", "trapezio"},
            "H22": {"prisma", "cilindro"},
        }
        for hab, sub in esperado.items():
            ids = {s["id"] for s in TAX["habilidades"][f"9º|{hab}"]["subtemas"]}
            self.assertTrue(sub <= ids, (hab, ids))

    def test_nao_inventa_subtema_incompativel(self):
        # malha quadriculada exclui círculo; "sem utilizar frações" não vira subtema fração
        self.assertNotIn("circulo", {s["id"] for s in TAX["habilidades"]["5º|H16"]["subtemas"]})
        self.assertNotIn("fracao", {s["id"] for s in TAX["habilidades"]["5º|H20"]["subtemas"]})
        # tabela (5º H21) não recebe gráficos; gráfico (5º H22) não recebe tabela
        self.assertFalse(any(s["id"].startswith("grafico") for s in TAX["habilidades"]["5º|H21"]["subtemas"]))
        self.assertFalse(any(s["id"].startswith("tabela") for s in TAX["habilidades"]["5º|H22"]["subtemas"]))


class TestPlanejador(unittest.TestCase):
    def _checa_regras(self, ano, hab, n):
        p = plano(ano, hab, n)
        k = K(ano, hab)
        self.assertEqual(len(p), n)
        c = Counter(s["subtema"] for s in p)
        self.assertEqual(len(c), min(n, k))
        self.assertLessEqual(max(c.values()), math.ceil(n / k))
        minimos = {1: 1, 2: min(2, k), 3: min(2, k), 5: min(3, k)}
        if n in minimos:
            self.assertGreaterEqual(len(c), minimos[n])
        return p

    def test_tamanhos_de_lote_varios_k(self):
        for ano, hab in [("9º", "H16"), ("9º", "H17"), ("9º", "H18"), ("9º", "H20"), ("9º", "H21"),
                         ("9º", "H22"), ("5º", "H16"), ("5º", "H17"), ("5º", "H18"), ("5º", "H20"),
                         ("5º", "H21"), ("5º", "H22"), ("9º", "H24"), ("2º", "H17")]:
            for n in (1, 2, 3, 5, 10, 20):
                with self.subTest(ano=ano, hab=hab, n=n):
                    self._checa_regras(ano, hab, n)

    def test_muitos_subtemas(self):
        self.assertGreaterEqual(K("9º", "H24"), 8)
        p = self._checa_regras("9º", "H24", 5)
        self.assertEqual(len({s["subtema"] for s in p}), 5)

    def test_poucos_subtemas(self):
        self.assertEqual(K("9º", "H22"), 2)
        p = self._checa_regras("9º", "H22", 10)
        self.assertEqual(Counter(s["subtema"] for s in p), Counter({"prisma": 5, "cilindro": 5}))

    def test_um_subtema_diversifica_outros_eixos(self):
        self.assertEqual(K("2º", "H17"), 1)
        p = plano("2º", "H17", 5)
        self.assertEqual(len({s["subtema"] for s in p}), 1)
        self.assertGreaterEqual(len({s["tipo_raciocinio"] for s in p}), 2)
        self.assertEqual(len({s["contexto"] for s in p}), 5)
        self.assertGreaterEqual(len({s["estrutura"] for s in p}), 4)

    def test_h17_9_n10_nao_so_triangulos(self):
        p = plano("9º", "H17", 10)
        c = Counter(s["subtema"] for s in p)
        self.assertEqual(c["triangulo"], 5)
        self.assertEqual(c["quadrilatero"], 5)
        # raciocínio também alterna (lados/ângulos) dentro de cada subtema
        for sub in ("triangulo", "quadrilatero"):
            self.assertEqual(len({s["tipo_raciocinio"] for s in p if s["subtema"] == sub}), 2)

    def test_determinismo_por_seed(self):
        self.assertEqual(plano("9º", "H16", 10, seed=7), plano("9º", "H16", 10, seed=7))
        self.assertNotEqual(plano("9º", "H16", 10, seed=7), plano("9º", "H16", 10, seed=8))

    def test_historico_prioriza_subtema_menos_usado(self):
        hist = [{"subtema": "triangulo", "contexto": "feira"}] * 3
        p = plano("9º", "H17", 1, historico=hist)
        self.assertEqual(p[0]["subtema"], "quadrilatero")
        self.assertNotEqual(p[0]["contexto"], "feira")

    def test_sufixo_prompt_formato_unico(self):
        s = plano("9º", "H17", 1)[0]
        suf = dv.sufixo_prompt(s)
        self.assertRegex(suf, r"^ Subtema: [^.]+\. Tipo de raciocínio: [^.]+\. Contexto: [^.]+\.$")


class TestClassificador(unittest.TestCase):
    def test_h17_9_triangulo_vs_quadrilatero(self):
        t = q("Um triângulo tem os três lados com medidas diferentes. Como ele é classificado quanto aos lados?",
              ("Escaleno", "Isósceles", "Equilátero", "Retângulo", "Nenhuma das alternativas anteriores"))
        c = dv.classificar_questao(t, "9º", "H17", taxonomia=TAX)
        self.assertEqual(c["subtema"], "triangulo")
        self.assertEqual(c["tipo_raciocinio"], "classificacao_lados")
        u = q("Um quadrilátero tem dois pares de lados paralelos e quatro ângulos retos, mas lados de medidas "
              "diferentes. Qual é o nome desse quadrilátero?",
              ("Quadrado", "Retângulo", "Losango", "Trapézio", "Nenhuma das alternativas anteriores"))
        c2 = dv.classificar_questao(u, "9º", "H17", taxonomia=TAX)
        self.assertEqual(c2["subtema"], "quadrilatero")

    def test_triangulo_retangulo_nao_vira_quadrilatero(self):
        c = dv.classificar_questao("Um triângulo retângulo tem um ângulo de 90 graus.", "9º", "H17", taxonomia=TAX)
        self.assertEqual(c["subtema"], "triangulo")

    def test_campos_e_indefinidos(self):
        c = dv.classificar_questao("xyz", "9º", "H17", taxonomia=TAX)
        self.assertEqual(set(c), {"subtema", "tipo_raciocinio", "objeto_matematico", "contexto",
                                  "estrutura", "representacao"})
        self.assertEqual(c["subtema"], "outros")
        self.assertEqual(c["contexto"], dv.SEM_CONTEXTO)

    def test_geometria_9_e_5(self):
        casos = [
            ("9º", "H16", "A soma dos ângulos internos de um triângulo é 180°. Se dois ângulos medem 50° e 60°, "
                          "quanto mede o terceiro ângulo?", "soma_angulos_internos"),
            ("9º", "H16", "É possível formar um triângulo com lados de 2 cm, 3 cm e 7 cm?", "condicao_existencia"),
            ("9º", "H18", "Uma corda de uma circunferência passa pelo centro. Qual o nome dessa corda?", "corda"),
            ("9º", "H20", "Qual é o perímetro de um terreno retangular de 20 m por 10 m?", "retangulo_quadrado"),
            ("9º", "H21", "Qual é a área de um trapézio de bases 8 cm e 4 cm e altura 3 cm?", "trapezio"),
            ("9º", "H22", "Uma lata tem formato de cilindro com raio 5 cm e altura 10 cm. Qual o volume?", "cilindro"),
            ("5º", "H17", "O relógio digital marca 14:30. Que horas são?",
             "relogio_digital"),
            ("5º", "H18", "O filme começou às 15h e durou 2 horas. Quanto tempo durou a sessão?", "duracao"),
            ("5º", "H20", "Em uma urna há 5 bolas azuis e 1 vermelha. Qual cor tem mais chance de sair?",
             "chance_maior_menor"),
            ("5º", "H21", "A tabela de dupla entrada mostra meninos e meninas por turma.", "tabela_dupla"),
            ("5º", "H22", "O gráfico de colunas mostra as vendas da semana.", "grafico_colunas"),
            ("5º", "H16", "Um quadrado desenhado na malha quadriculada ocupa quantos quadradinhos?",
             "retangulo_quadrado"),
        ]
        for ano, hab, txt, esperado in casos:
            with self.subTest(ano=ano, hab=hab):
                self.assertEqual(dv.classificar_questao(txt, ano, hab, taxonomia=TAX)["subtema"], esperado)


class TestMetricas(unittest.TestCase):
    A = q("Uma loja vende uma bicicleta por R$ 200,00 com 10% de desconto. Qual o preço final?")
    B = q("Uma loja vende uma bicicleta por R$ 350,00 com 25% de desconto. Qual o preço final?")
    C = q("Um quadrado tem lado de 4 cm. Quanto mede cada ângulo interno desse quadrilátero?")

    def test_troca_de_numeros_e_duplicata(self):
        self.assertTrue(dv.e_near_duplicata(self.A, self.B))
        self.assertFalse(dv.e_near_duplicata(self.A, self.C))
        self.assertAlmostEqual(dv.duplicate_rate([self.A, self.B, self.C]), 1 / 3)

    def test_similaridade(self):
        s = dv.semantic_similarity([self.A, self.B, self.C])
        self.assertAlmostEqual(s["cosseno_max"], 1.0)
        self.assertLess(s["cosseno_media"], 1.0)

    def test_componentes_do_diversity_score(self):
        r = dv.diversity_score([self.A, self.B, self.C], "9º", "H17", taxonomia=TAX)
        for chave in ("subtema_distribution", "coverage_score", "duplicate_rate", "semantic_similarity",
                      "structural_diversity", "context_diversity", "raciocinio_diversity", "pesos",
                      "componentes", "diversity_score"):
            self.assertIn(chave, r)
        self.assertEqual(set(r["componentes"]), set(dv.PESOS_DIVERSIDADE))
        self.assertTrue(0.0 <= r["diversity_score"] <= 1.0)

    def test_lote_diverso_pontua_mais(self):
        tri = q("Um triângulo tem lados 3 cm, 3 cm e 3 cm. Como ele é classificado quanto aos lados?")
        qua = q("Na horta, um canteiro é um quadrilátero com quatro ângulos retos e lados diferentes. "
                "Como ele é classificado?")
        tri2 = q("Um triângulo tem lados 5 cm, 5 cm e 5 cm. Como ele é classificado quanto aos lados?")
        bom = dv.diversity_score([tri, qua], "9º", "H17", taxonomia=TAX)
        ruim = dv.diversity_score([tri, tri2], "9º", "H17", taxonomia=TAX)
        self.assertEqual(bom["coverage_score"], 1.0)
        self.assertEqual(ruim["coverage_score"], 0.5)
        self.assertEqual(ruim["duplicate_rate"], 0.5)
        self.assertGreater(bom["diversity_score"], ruim["diversity_score"])

    def test_coverage_score(self):
        self.assertEqual(dv.coverage_score(["a", "b", "a"], 2), 1.0)
        self.assertEqual(dv.coverage_score(["a", "a", "a"], 5), 1 / 3)


class TestRegeneracao(unittest.TestCase):
    TRI = [q(f"Um triângulo tem lados {a} cm, {b} cm e {c} cm. Como ele é classificado quanto aos lados?")
           for a, b, c in ((3, 4, 5), (2, 2, 3))]

    def test_violacoes_e_restricao(self):
        slot = plano("9º", "H17", 4)[0]
        slot = dict(slot, subtema="quadrilatero", subtema_rotulo="quadriláteros")
        cand = q("Um triângulo tem lados 6 cm, 6 cm e 6 cm. Como ele é classificado quanto aos lados?")
        v = dv.violacoes_diversidade(self.TRI, cand, slot, "9º", "H17", quantidade=4, taxonomia=TAX)
        tipos = {x["tipo"] for x in v}
        self.assertIn("subtema_fora_do_plano", tipos)
        self.assertIn("subtema_saturado", tipos)
        self.assertIn("near_duplicata", tipos)
        txt = dv.montar_restricao(v, "H17", slot)
        self.assertIn("O lote já contém 2 questão(ões) sobre triângulos", txt)
        self.assertIn("H17: quadriláteros", txt)

    def test_candidato_ok_sem_violacao(self):
        slot = dict(plano("9º", "H17", 3)[0], subtema="quadrilatero", subtema_rotulo="quadriláteros")
        cand = q("Um quadrilátero tem quatro lados iguais e quatro ângulos retos. Qual é o seu nome?")
        self.assertEqual(dv.violacoes_diversidade(self.TRI, cand, slot, "9º", "H17", 3, taxonomia=TAX), [])

    def test_gerar_lote_diverso_regenera_com_limite(self):
        chamadas = []

        def gerar(slot, restricao, tentativa):
            chamadas.append((slot["subtema"], restricao))
            # "modelo" viciado em triângulos até receber restrição
            if slot["subtema"] == "quadrilatero" and restricao:
                return q(f"Um quadrilátero {slot['indice']} tem lados paralelos dois a dois. Como classificá-lo?")
            return q(f"Um triângulo tem lados {slot['indice']} cm, 4 cm e 5 cm. Como classificá-lo quanto aos lados?")

        aceitas, rel = dv.gerar_lote_diverso(gerar, "9º", "H17", 4, "Moderado", seed=1, max_tentativas=2,
                                             taxonomia=TAX)
        self.assertEqual(len(aceitas), 4)
        subs = Counter(dv.classificar_questao(a, "9º", "H17", taxonomia=TAX)["subtema"] for a in aceitas)
        self.assertEqual(subs["quadrilatero"], 2)
        self.assertTrue(all(r["tentativas"] <= 3 for r in rel))


if __name__ == "__main__":
    unittest.main()
