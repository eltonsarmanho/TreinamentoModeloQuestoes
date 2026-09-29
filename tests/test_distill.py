"""Testes da destilação com diversidade (sem rede: cliente falso)."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import distill_teacher as dt  # noqa: E402
import diversidade  # noqa: E402

DESCR_H17 = "Classificar triângulos ou quadriláteros em relação aos lados ou ângulos internos."


def questao(enunciado, alts=None, correta="A"):
    alts = alts or {"A": "Isósceles", "B": "Escaleno", "C": "Equilátero",
                    "D": "Retângulo", "E": "Nenhuma das alternativas anteriores"}
    return {"enunciado": enunciado, "alternativas": alts,
            "resolucao_passo_a_passo": "Dois lados são iguais, logo o triângulo é isósceles.",
            "resposta_correta": correta, "difficulty": "MEDIUM"}


Q_TRI = questao("Um triângulo tem lados de 5 cm, 5 cm e 7 cm. Como ele é classificado quanto aos lados?")
Q_QUAD = questao("Um quadrilátero tem quatro lados iguais e quatro ângulos retos. Que quadrilátero é esse?",
                 {"A": "Quadrado", "B": "Trapézio", "C": "Losango", "D": "Paralelogramo",
                  "E": "Nenhuma das alternativas anteriores"})


class ClienteFalso:
    """Imita InferenceClient.chat_completion devolvendo respostas em fila."""
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.prompts = []

    def chat_completion(self, messages, **kw):
        self.prompts.append(messages[1]["content"])
        texto = self.respostas.pop(0) if self.respostas else ""
        msg = types.SimpleNamespace(content=texto)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def slot_9h17(subtema):
    hab = diversidade.obter_habilidade("9º", "H17")
    s = next(x for x in hab["subtemas"] if x["id"] == subtema)
    return {"indice": 0, "subtema": subtema, "subtema_rotulo": s["rotulo"],
            "tipo_raciocinio": hab["tipos_raciocinio"][0], "tipo_raciocinio_rotulo": "x",
            "contexto": "feira", "contexto_rotulo": "feira livre", "estrutura": "pergunta_direta",
            "estrutura_rotulo": "p", "dificuldade": "Moderado"}


class TestPlano(unittest.TestCase):
    def test_plano_cobre_subtemas_9h17(self):
        itens = [{"ano": "9º", "habilidade": "H17", "descricao": DESCR_H17, "dificuldade": "Moderado",
                  "quantidade": 4, "subtema": None, "tipo_raciocinio": None}]
        plano = dt.planejar_itens(itens, seed=42)
        subs = {s["subtema"] for _, s in plano}
        self.assertEqual(len(plano), 4)
        self.assertEqual(subs, {"triangulo", "quadrilatero"})

    def test_plano_deterministico(self):
        itens = dt.itens_de_tuplas([("5º", "H18", "d", "Fácil")], 5)
        a = dt.planejar_itens(itens, seed=7)
        b = dt.planejar_itens(itens, seed=7)
        self.assertEqual([s for _, s in a], [s for _, s in b])

    def test_rotacao_continua_entre_itens_mesma_habilidade(self):
        itens = dt.itens_de_tuplas([("9º", "H17", DESCR_H17, "Fácil"), ("9º", "H17", DESCR_H17, "Difícil")], 1)
        plano = dt.planejar_itens(itens, seed=1)
        self.assertEqual(len({s["subtema"] for _, s in plano}), 2)

    def test_carregar_plano_arquivo_e_fixacao(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plano.json"
            p.write_text(json.dumps({"itens": [{"ano": "9º", "habilidade": "H17", "dificuldade": "Fácil",
                                                 "subtema": "quadrilatero", "quantidade": 2}]}))
            itens = dt.carregar_plano(p, tuples=[("9º", "H17", DESCR_H17, "Fácil")])
        self.assertEqual(itens[0]["descricao"], DESCR_H17)
        plano = dt.planejar_itens(itens)
        self.assertTrue(all(s["subtema"] == "quadrilatero" for _, s in plano))

    def test_formato_curador_sem_dificuldade(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plano.json"
            p.write_text(json.dumps({"itens": [{"ano": "9º", "habilidade": "H17", "descricao": DESCR_H17,
                                                 "subtema": "triangulo", "quantidade": 4,
                                                 "tipos_raciocinio": ["classificacao_angulos"]}]}))
            itens = dt.carregar_plano(p, tuples=[("9º", "H17", DESCR_H17, "Fácil"),
                                                 ("9º", "H17", DESCR_H17, "Difícil")])
        plano = dt.planejar_itens(itens)
        self.assertEqual({it["dificuldade"] for it, _ in plano}, {"Fácil", "Difícil"})
        self.assertTrue(all(s["tipo_raciocinio"] == "classificacao_angulos" for _, s in plano))

    def test_prompt_treino_tem_sufixo_unico_e_professor_tem_aderencia(self):
        slot = slot_9h17("triangulo")
        treino = dt.prompt_usuario("9º", "H17", DESCR_H17, "Moderado", slot)
        prof = dt.prompt_professor("9º", "H17", DESCR_H17, "Moderado", slot)
        self.assertTrue(treino.endswith(diversidade.sufixo_prompt(slot)))
        self.assertTrue(prof.startswith(treino))
        self.assertIn("Não troque de habilidade", prof)
        self.assertNotIn("Não troque", treino)


class TestFiltros(unittest.TestCase):
    def test_subtema_divergente(self):
        wrapper = {"questoes": [Q_QUAD]}
        self.assertEqual(dt.filtrar(wrapper, "", set(), slot=slot_9h17("triangulo"),
                                    ano="9º", habilidade="H17", aceitas={}), "subtema_divergente")
        self.assertIsNone(dt.filtrar(wrapper, "", set(), slot=slot_9h17("quadrilatero"),
                                     ano="9º", habilidade="H17", aceitas={}))

    def test_sem_casamento_nao_rejeita(self):
        q = questao("Qual das opções abaixo está correta para esta classificação?")
        self.assertFalse(dt.subtema_divergente(q, "9º", "H17", slot_9h17("triangulo")))

    def test_near_duplicata_troca_numeros(self):
        aceitas = {}
        self.assertIsNone(dt.filtrar({"questoes": [Q_TRI]}, "", set(), slot=slot_9h17("triangulo"),
                                     ano="9º", habilidade="H17", aceitas=aceitas))
        q2 = dict(Q_TRI, enunciado=Q_TRI["enunciado"].replace("5 cm, 5 cm e 7 cm", "8 cm, 8 cm e 3 cm"))
        self.assertEqual(dt.filtrar({"questoes": [q2]}, "", set(), slot=slot_9h17("triangulo"),
                                    ano="9º", habilidade="H17", aceitas=aceitas), "near_duplicata")

    def test_near_dup_isolada_por_ano(self):
        aceitas = {("5º", "H17"): [Q_TRI]}
        self.assertIsNone(dt.filtrar({"questoes": [Q_TRI]}, "", set(), slot=slot_9h17("triangulo"),
                                     ano="9º", habilidade="H17", aceitas=aceitas))

    def test_filtros_antigos_mantidos(self):
        self.assertEqual(dt.filtrar(None, "", set()), "json_invalido")
        self.assertEqual(dt.filtrar({"questoes": [{"enunciado": "x", "resposta_correta": "Z"}]}, "", set()), "estrutura")
        fig = questao("Observe a figura abaixo e classifique o triângulo.")
        self.assertEqual(dt.filtrar({"questoes": [fig]}, "", set()), "menciona_figura")
        vistos = set()
        self.assertIsNone(dt.filtrar({"questoes": [Q_TRI]}, "", vistos))
        self.assertEqual(dt.filtrar({"questoes": [Q_TRI]}, "", vistos), "duplicata")

    def test_build_example_meta(self):
        slot = slot_9h17("triangulo")
        ex = dt.build_example("9º", "H17", DESCR_H17, "Moderado", Q_TRI, 1, "prof", slot)
        self.assertEqual(ex["meta"]["subtema"], "triangulo")
        self.assertEqual(ex["meta"]["contexto"], "feira")
        self.assertIn("tipo_raciocinio", ex["meta"])
        self.assertIn(" Subtema: ", ex["messages"][1]["content"])
        self.assertEqual(set(json.loads(ex["messages"][2]["content"])), {"questoes"})


class TestGerarComClienteFalso(unittest.TestCase):
    def test_gerar_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            orig = (dt.DISTILL_PATH, dt.TRAIN_PATH, dt.load_tuples)
            dt.DISTILL_PATH, dt.TRAIN_PATH = d / "distill.jsonl", d / "train.jsonl"
            dt.load_tuples = lambda limit=None: [("9º", "H17", DESCR_H17, "Moderado")]
            try:
                args = types.SimpleNamespace(plano=None, per_tuple=2, limit_tuples=None, seed=42,
                                             teacher="falso", provider="x", max_tokens=10,
                                             temperature=0.9, strict=False, max_errors=3, retry_wait=0)
                plano = dt.planejar_itens(dt.itens_de_tuplas(dt.load_tuples(), 2), seed=42)
                alvo = [s["subtema"] for _, s in plano]
                resp = [json.dumps({"questoes": [Q_TRI if a == "triangulo" else Q_QUAD]}) for a in alvo]
                cli = ClienteFalso(resp)
                r = dt.gerar(args, client=cli)
                linhas = [json.loads(l) for l in open(dt.DISTILL_PATH)]
            finally:
                dt.DISTILL_PATH, dt.TRAIN_PATH, dt.load_tuples = orig
        self.assertEqual(r["aceitos"], 2)
        self.assertEqual(sorted(e["meta"]["subtema"] for e in linhas), sorted(alvo))
        self.assertTrue(all("Não troque de habilidade" in p for p in cli.prompts))


if __name__ == "__main__":
    unittest.main()
