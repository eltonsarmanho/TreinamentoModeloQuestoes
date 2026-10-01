"""
Invariantes do conjunto FIXO de prompts de avaliação (data/prompts_diversidade.json).

Este arquivo é a única entrada de `avaliar_diversidade.py`: se um prompt entrar
inválido, a rodada custa horas de CPU e só se descobre no fim. Os testes abaixo
travam, de forma determinística e sem tocar no modelo, tudo que dá para verificar
antes de gerar: campos, unicidade de id/seed, existência do par (ano, habilidade)
na taxonomia, coerência de k_subtemas com a taxonomia real, dificuldade dentro de
graus_resolucao, e o fato de que o plano de subtemas realmente monta.

Também congela o PREFIXO v1 (P01..P16). `carregar_prompts` corta do INÍCIO
(`prompts[:num_prompts]`), e os relatórios já gravados em outputs/diversidade_*.json
são pareados por id+seed: qualquer edição, reordenação ou inserção antes de P16
quebra a comparação histórica. Conjunto novo só cresce por ANEXAÇÃO ao fim.

Compatível com unittest e pytest:
    venv/bin/python -m pytest tests/test_prompts_diversidade.py -q
"""
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import avaliar_diversidade as av  # noqa: E402
import diversidade as dv  # noqa: E402

DADOS = json.loads((ROOT / "data" / "prompts_diversidade.json").read_text(encoding="utf-8"))
PROMPTS = DADOS["prompts"]
TAX = dv.carregar_taxonomia()
HAB = TAX["habilidades"]

CAMPOS = ("id", "ano", "habilidade", "descricao", "dificuldade",
          "quantidade", "seed", "k_subtemas")

# P01..P16 congelados: sha256 do JSON canônico dos 16 primeiros prompts.
# NÃO atualizar este hash para "consertar" o teste — ele existe justamente para
# impedir que o prefixo mude. Prompt novo entra DEPOIS do último.
PREFIXO_V1_SHA256 = "019beea5af3af4cf17989f6ca9d1dd0081b35d97a803a5bfebda1b2eb561ad51"
N_PREFIXO_V1 = 16

# Aderência à habilidade só discrimina com K>=2 subtemas (avaliar_diversidade
# define aderencia_mensuravel = K>=2). Estes quatro entraram com K=1 por decisão
# explícita — P14/P15/P16 vêm do conjunto v1 e P30 (9º|H12) entrou na v2 pela
# verificabilidade medida (~50%), não pela aderência. Qualquer OUTRO prompt com
# K=1 é erro: ou a habilidade está errada, ou falta subtema na taxonomia.
K1_PERMITIDOS = {"P14-9º-H12-N1", "P15-2º-H17-N3", "P16-9º-H08-N5", "P30-9º-H12-N3"}

RE_ID = re.compile(r"^P(\d{2})-(\S+?)-([A-Z]+\d+)-N(\d+)$")


def chave(p):
    return f"{p['ano']}|{p['habilidade']}"


class TestEstruturaDoArquivo(unittest.TestCase):
    def test_carrega_pelo_runtime(self):
        """O teste e avaliar_diversidade.py leem o mesmo conjunto."""
        self.assertEqual(av.carregar_prompts(), PROMPTS)
        self.assertGreaterEqual(len(PROMPTS), 32)

    def test_campos_obrigatorios_e_tipos(self):
        for p in PROMPTS:
            with self.subTest(id=p.get("id")):
                self.assertEqual(tuple(p.keys()), CAMPOS,
                                 "campos ausentes, extras ou fora de ordem")
                for c in ("id", "ano", "habilidade", "descricao", "dificuldade"):
                    self.assertIsInstance(p[c], str)
                    self.assertTrue(p[c].strip(), f"{c} vazio")
                for c in ("quantidade", "seed", "k_subtemas"):
                    self.assertIsInstance(p[c], int)
                    self.assertFalse(isinstance(p[c], bool))
                self.assertGreater(p["quantidade"], 0)
                self.assertGreaterEqual(p["seed"], 0)

    def test_ids_e_seeds_unicos(self):
        ids = [p["id"] for p in PROMPTS]
        seeds = [p["seed"] for p in PROMPTS]
        self.assertEqual(len(set(ids)), len(ids), "id repetido")
        # seeds repetidas colapsariam duas linhas do relatório no pareamento id+seed
        self.assertEqual(len(set(seeds)), len(seeds), "seed repetida")

    def test_id_codifica_ano_habilidade_quantidade_e_seed(self):
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                m = RE_ID.match(p["id"])
                self.assertIsNotNone(m, "id fora do padrão Pnn-<ano>-<HAB>-N<qtd>")
                i, ano, hab, n = m.groups()
                self.assertEqual(ano, p["ano"])
                self.assertEqual(hab, p["habilidade"])
                self.assertEqual(int(n), p["quantidade"])
                self.assertEqual(int(i), p["seed"], "prefixo numérico do id != seed")

    def test_versao_e_nota_presentes(self):
        self.assertGreaterEqual(DADOS["versao"], 2)
        self.assertTrue(DADOS["nota"].strip())


class TestPrefixoCongelado(unittest.TestCase):
    """O conjunto só cresce por anexação; o prefixo v1 é imutável."""

    def test_prefixo_v1_intacto(self):
        prefixo = PROMPTS[:N_PREFIXO_V1]
        canon = json.dumps(prefixo, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canon.encode("utf-8")).hexdigest(), PREFIXO_V1_SHA256,
                         "P01..P16 mudaram: os relatórios já rodados deixam de ser pareáveis")

    def test_num_prompts_corta_do_inicio_e_devolve_o_conjunto_v1(self):
        self.assertEqual(av.carregar_prompts(num_prompts=N_PREFIXO_V1), PROMPTS[:N_PREFIXO_V1])

    def test_primeiro_prompt_e_o_obrigatorio(self):
        p = PROMPTS[0]
        self.assertEqual((p["ano"], p["habilidade"], p["quantidade"]), ("9º", "H17", 10))


class TestCoerenciaComATaxonomia(unittest.TestCase):
    def test_par_ano_habilidade_existe(self):
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                self.assertIn(chave(p), HAB,
                              "sem taxonomia: planejar_lote levantaria KeyError na rodada")

    def test_k_subtemas_bate_com_a_taxonomia(self):
        """k_subtemas é documentação (metricas_lote recalcula K), mas mentir nele
        engana quem lê o conjunto ao planejar orçamento de rodada."""
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                self.assertEqual(p["k_subtemas"], len(HAB[chave(p)]["subtemas"]))

    def test_aderencia_mensuravel_exceto_excecoes_declaradas(self):
        k1 = {p["id"] for p in PROMPTS if len(HAB[chave(p)]["subtemas"]) < 2}
        self.assertEqual(k1, K1_PERMITIDOS,
                         "prompt com K=1 fora da lista: aderência não discrimina "
                         "(aderentes/quantidade = 100% por construção)")
        mensuraveis = len(PROMPTS) - len(k1)
        self.assertGreaterEqual(mensuraveis, len(PROMPTS) - len(K1_PERMITIDOS))

    def test_dificuldade_pertence_aos_graus_da_habilidade(self):
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                self.assertIn(p["dificuldade"], av.DIFFICULTY_MAP,
                              "dificuldade não mapeia para EASY/MEDIUM/HARD")
                self.assertIn(p["dificuldade"], HAB[chave(p)]["graus_resolucao"],
                              "dificuldade sem lastro no banco para essa habilidade")

    def test_descricao_e_uma_das_formas_usadas_no_projeto(self):
        """A descrição vai literal no USER_TEMPLATE. Se divergir das formas vistas
        no treino, o prompt sai fora da distribuição do fine-tuning."""
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                self.assertIn(p["descricao"], HAB[chave(p)]["descricoes"])


class TestPlanejamentoDeLote(unittest.TestCase):
    def test_planejar_lote_roda_para_todos(self):
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                plano = dv.planejar_lote(p["ano"], p["habilidade"], p["quantidade"],
                                         p["dificuldade"], p["seed"], taxonomia=TAX)
                self.assertEqual(len(plano), p["quantidade"])

    def test_plano_cobre_min_n_k_subtemas_distintos(self):
        for p in PROMPTS:
            with self.subTest(id=p["id"]):
                plano = dv.planejar_lote(p["ano"], p["habilidade"], p["quantidade"],
                                         p["dificuldade"], p["seed"], taxonomia=TAX)
                distintos = {s["subtema"] for s in plano}
                self.assertEqual(len(distintos),
                                 min(p["quantidade"], len(HAB[chave(p)]["subtemas"])))

    def test_plano_e_deterministico(self):
        p = PROMPTS[-1]
        a = dv.planejar_lote(p["ano"], p["habilidade"], p["quantidade"],
                             p["dificuldade"], p["seed"], taxonomia=TAX)
        b = dv.planejar_lote(p["ano"], p["habilidade"], p["quantidade"],
                             p["dificuldade"], p["seed"], taxonomia=TAX)
        self.assertEqual(a, b)


class TestOrcamentoDaRodada(unittest.TestCase):
    def test_soma_de_questoes_por_rodada(self):
        """Serve de trava de custo: 152 questões/rodada a ~15 s = ~38 min por
        modo. Subir esse número é decisão de orçamento, não efeito colateral."""
        self.assertEqual(sum(p["quantidade"] for p in PROMPTS), 152)

    def test_metade_nova_nao_desequilibra_o_orcamento(self):
        antigos = sum(p["quantidade"] for p in PROMPTS[:N_PREFIXO_V1])
        novos = sum(p["quantidade"] for p in PROMPTS[N_PREFIXO_V1:])
        self.assertEqual(antigos, novos)

    def test_cobertura_de_anos_e_de_n(self):
        self.assertEqual({p["ano"] for p in PROMPTS}, {"2º", "5º", "9º"})
        self.assertEqual({p["quantidade"] for p in PROMPTS}, {1, 3, 5, 10})


if __name__ == "__main__":
    unittest.main()
