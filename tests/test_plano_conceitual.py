"""
Correções do PLANO para habilidades conceituais (item 2, 2026-10-01).

Motivação: teste real de 2026-10-01 com o modelo promovido, 20 questões de
9º H17 ("Classificar triângulos ou quadriláteros em relação aos lados ou aos
ângulos internos"), Fácil, modo planejado. A auditoria humana
(outputs/testes_locais/Log.txt) aprovou só 5/20. Por tipo de slot: triângulo x
lados 4/5, triângulo x ângulos 1/5, quadrilátero x lados 0/5, quadrilátero x
ângulos 0/5. Este arquivo cobre:
  (a) compatibilidade subtema x tipo de raciocínio na taxonomia;
  (b) estrutura 'cálculo' fora das habilidades conceituais;
  (c) o sufixo nomeia o eixo da classificação (no FIM, formato treinado intacto);
  (d) dados_insuficientes_classificacao (ônibus / garrafas PET);
  (e) SEM_CONTEXTO como contexto planejado (opção, DESLIGADA por padrão).
Os enunciados dos casos reais estão copiados aqui (outputs/ não é versionado).
    venv/bin/python -m pytest tests/test_plano_conceitual.py -q
"""
import copy
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_taxonomia as bt  # noqa: E402
import diversidade as dv  # noqa: E402

TAX = bt.construir()
TAX_SEMCTX = bt.construir(sem_contexto_conceitual=True)
CONCEITUAIS = {"2º|H11", "2º|H12", "5º|H12", "5º|H13", "5º|H14", "9º|H14", "9º|H17"}
DIFS = ("Fácil", "Moderado", "Difícil")
TAMANHOS = (1, 2, 3, 5, 10, 20)
CLASSIF = {"classificacao_lados", "classificacao_angulos", "classificacao_propriedades"}


def planos(tax, chave, seeds=range(8)):
    ano, hab = chave.split("|")
    for seed in seeds:
        for dif in DIFS:
            for n in TAMANHOS:
                yield n, dv.planejar_lote(ano, hab, n, dif, seed, taxonomia=tax)


def q(enunciado, alts=("a", "b", "c", "d", "Nenhuma das alternativas anteriores")):
    return {"enunciado": enunciado, "alternativas": dict(zip("ABCDE", alts)),
            "resolucao_passo_a_passo": "", "resposta_correta": "A", "difficulty": "EASY"}


def tax_com_overrides(ov, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ov.json"
        p.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
        return bt.construir(overrides_path=p, **kw)


def _sem_chaves_novas(tax):
    """Cópia da taxonomia sem nada do que o item 2 acrescentou por habilidade."""
    t = copy.deepcopy(tax)
    for h in t["habilidades"].values():
        for c in ("conceitual", "estruturas", "contextos"):
            h.pop(c, None)
        for s in h["subtemas"]:
            s["tipos_raciocinio"] = list(h["tipos_raciocinio"])
    return t


# --------------------------------------------------------------------------
# (a) compatibilidade subtema x tipo
# --------------------------------------------------------------------------
class TestCompatibilidade(unittest.TestCase):
    def test_h17_quadrilatero_so_propriedades_triangulo_dois_eixos(self):
        # Os 3 itens reais de quadrilátero (MT9050, MT9082, MT9084) nomeiam a
        # figura combinando lados, paralelismo e ângulos retos; nenhum
        # classifica quadrilátero só por ângulos ou só por lados.
        for tax in (TAX, dv.carregar_taxonomia()):  # build e JSON em disco
            sub = {s["id"]: s["tipos_raciocinio"] for s in tax["habilidades"]["9º|H17"]["subtemas"]}
            self.assertEqual(sub["triangulo"], ["classificacao_lados", "classificacao_angulos"])
            self.assertEqual(sub["quadrilatero"], ["classificacao_propriedades"])

    def test_nenhum_slot_com_combinacao_invalida_em_nenhuma_habilidade(self):
        for chave, h in TAX["habilidades"].items():
            compat = {s["id"]: set(s["tipos_raciocinio"]) for s in h["subtemas"]}
            for n, p in planos(TAX, chave, seeds=range(3)):
                for s in p:
                    self.assertIn(s["tipo_raciocinio"], compat[s["subtema"]], (chave, n, s))
                    self.assertTrue(dv.tipo_aplica_a_subtema(s["tipo_raciocinio"], s["subtema"]))

    def test_h17_nunca_quadrilatero_por_lados_ou_angulos(self):
        # R1/R2: 36/150 slots de quadrilátero x ângulos no plano antigo; 0/10
        # corretas nas células de quadrilátero.
        for n, p in planos(TAX, "9º|H17", seeds=range(30)):
            for s in p:
                if s["subtema"] == "quadrilatero":
                    self.assertEqual(s["tipo_raciocinio"], "classificacao_propriedades")
                else:
                    self.assertIn(s["tipo_raciocinio"], {"classificacao_lados", "classificacao_angulos"})

    def test_lote_de_3_cobre_os_3_tipos(self):
        # Com N > K, o subtema que repete é o que admite mais tipos: lote de 3
        # vira triângulo, quadrilátero, triângulo e cobre lados, ângulos e
        # propriedades (antes, quadrilátero-triângulo-quadrilátero cobria 2/3).
        for seed in range(40):
            for dif in DIFS:
                p = dv.planejar_lote("9º", "H17", 3, dif, seed, taxonomia=TAX)
                self.assertEqual({s["tipo_raciocinio"] for s in p}, CLASSIF)

    def test_lote_de_1_continua_sorteando_o_subtema(self):
        subs = {dv.planejar_lote("9º", "H17", 1, "Fácil", seed, taxonomia=TAX)[0]["subtema"]
                for seed in range(20)}
        self.assertEqual(subs, {"triangulo", "quadrilatero"})

    def test_outras_habilidades_tem_o_plano_de_antes(self):
        # Nenhuma habilidade fora das conceituais ganhou chave nova nem
        # restrição de tipo, e para elas o plano é igual ao calculado sem
        # nada do item 2 (mesmo subtema, tipo, contexto e estrutura).
        semnovo = _sem_chaves_novas(TAX)
        for chave, h in TAX["habilidades"].items():
            if chave in CONCEITUAIS:
                continue
            with self.subTest(chave=chave):
                self.assertFalse({"conceitual", "estruturas", "contextos"} & set(h))
                for s in h["subtemas"]:
                    self.assertEqual(s["tipos_raciocinio"], h["tipos_raciocinio"])
                ano, hab = chave.split("|")
                for n in (1, 3, 10):
                    self.assertEqual(dv.planejar_lote(ano, hab, n, "Moderado", 5, taxonomia=TAX),
                                     dv.planejar_lote(ano, hab, n, "Moderado", 5, taxonomia=semnovo))

    def test_conjunto_de_conceituais(self):
        self.assertEqual({k for k, h in TAX["habilidades"].items() if h.get("conceitual")}, CONCEITUAIS)

    def test_override_tipos_por_subtema_e_adicionar(self):
        tax = tax_com_overrides({
            "_nota": "chave de comentário é ignorada",
            "9º|H17": {"tipos_por_subtema": {"triangulo": ["classificacao_lados"]},
                       "adicionar": [{"id": "losango", "rotulo": "losangos", "palavras_chave": ["losang"],
                                      "tipos_raciocinio": ["classificacao_propriedades"]}]}})
        sub = {s["id"]: s["tipos_raciocinio"] for s in tax["habilidades"]["9º|H17"]["subtemas"]}
        self.assertEqual(sub["triangulo"], ["classificacao_lados"])
        self.assertEqual(sub["losango"], ["classificacao_propriedades"])
        # tipos da habilidade = os que algum subtema usa
        self.assertEqual(tax["habilidades"]["9º|H17"]["tipos_raciocinio"],
                         ["classificacao_lados", "classificacao_propriedades"])

    def test_classificador_escolhe_tipo_compativel_com_o_subtema(self):
        # MT9050: "dois pares de lados opostos paralelos e ângulos retos" — com o
        # tipo da habilidade inteira, "ângulos retos" puxaria 'ângulos'.
        quad = q("Um quadrilátero tem dois pares de lados opostos paralelos e ângulos retos. "
                 "Como podemos classificar esse quadrilátero?",
                 ("Trapézio", "Retângulo", "Losango", "Paralelogramo", "Nenhuma das alternativas anteriores"))
        c = dv.classificar_questao(quad, "9º", "H17", taxonomia=TAX)
        self.assertEqual((c["subtema"], c["tipo_raciocinio"]), ("quadrilatero", "classificacao_propriedades"))
        tri = q("Triângulo com ângulos de 90°, 45° e 45°. Como ele é classificado quanto aos ângulos?",
                ("Retângulo", "Obtusângulo", "Acutângulo", "Equilátero", "Nenhuma das alternativas anteriores"))
        c = dv.classificar_questao(tri, "9º", "H17", taxonomia=TAX)
        self.assertEqual((c["subtema"], c["tipo_raciocinio"]), ("triangulo", "classificacao_angulos"))


# --------------------------------------------------------------------------
# (b) estruturas
# --------------------------------------------------------------------------
class TestEstruturas(unittest.TestCase):
    def test_conceitual_nunca_recebe_calculo(self):
        # 9º H17 recebia 'cálculo' em 26/150 slots no plano antigo.
        for chave in CONCEITUAIS:
            self.assertNotIn("calculo", TAX["habilidades"][chave]["estruturas"])
            for _, p in planos(TAX, chave, seeds=range(5)):
                self.assertFalse([s for s in p if s["estrutura"] == "calculo"], chave)

    def test_procedimental_continua_com_calculo(self):
        ests = {s["estrutura"] for _, p in planos(TAX, "9º|H21", seeds=range(3)) for s in p}
        self.assertIn("calculo", ests)

    def test_sem_contexto_nunca_com_problema_contextualizado(self):
        for chave in CONCEITUAIS:
            for _, p in planos(TAX_SEMCTX, chave, seeds=range(5)):
                for s in p:
                    if s["contexto"] == dv.SEM_CONTEXTO:
                        self.assertNotEqual(s["estrutura"], "problema_contextualizado", (chave, s))


# --------------------------------------------------------------------------
# (c) sufixo com o eixo
# --------------------------------------------------------------------------
PREFIXO = re.compile(r"^ Subtema: [^.]+\. Tipo de raciocínio: [^.]+\. Contexto: [^.]+\.")


class TestSufixo(unittest.TestCase):
    def _slot(self, subtema, tipo):
        return {"subtema": subtema, "subtema_rotulo": subtema, "tipo_raciocinio": tipo,
                "tipo_raciocinio_rotulo": dv._rotulo(dv.TIPOS_RACIOCINIO, tipo),
                "contexto": "horta", "contexto_rotulo": "horta comunitária"}

    def test_eixo_nomeado_no_fim_sem_mexer_no_formato_treinado(self):
        casos = {("triangulo", "classificacao_lados"): "quanto aos lados",
                 ("triangulo", "classificacao_angulos"): "quanto aos ângulos",
                 ("quadrilatero", "classificacao_propriedades"): "lados paralelos"}
        for (sub, tipo), trecho in casos.items():
            slot = self._slot(sub, tipo)
            suf = dv.sufixo_prompt(slot)
            base = (f" Subtema: {sub}. Tipo de raciocínio: {slot['tipo_raciocinio_rotulo']}."
                    f" Contexto: horta comunitária.")
            self.assertTrue(suf.startswith(base), suf)
            self.assertTrue(suf[len(base):].startswith(" Eixo: "), suf)
            self.assertIn(trecho, suf[len(base):])

    def test_instrucao_pede_os_dados_minimos(self):
        # (d) no prompt: medidas dos lados / dos ângulos (somando 180°, contra
        # "dois ângulos retos" de R1 Q5) / propriedades do quadrilátero.
        self.assertIn("três lados", dv.sufixo_prompt(self._slot("triangulo", "classificacao_lados")))
        self.assertIn("180°", dv.sufixo_prompt(self._slot("triangulo", "classificacao_angulos")))
        suf = dv.sufixo_prompt(self._slot("quadrilatero", "classificacao_propriedades"))
        self.assertIn("ângulos retos", suf)
        self.assertIn("mais específico", suf)

    def test_sem_instrucao_fora_da_classificacao_ou_em_subtema_incompativel(self):
        self.assertNotIn("Eixo:", dv.sufixo_prompt(self._slot("adicao", "resolucao_problema")))
        # slot antigo (ex.: curar_diversidade sobre uma questão já existente):
        # instrução de triângulo nunca vai para quadrilátero
        self.assertNotIn("Eixo:", dv.sufixo_prompt(self._slot("quadrilatero", "classificacao_lados")))

    def test_todo_slot_planejado_de_h17_tem_eixo(self):
        for _, p in planos(TAX, "9º|H17", seeds=range(3)):
            for s in p:
                suf = dv.sufixo_prompt(s)
                self.assertRegex(suf, PREFIXO)
                self.assertIn(" Eixo: ", suf)

    def test_outras_habilidades_sufixo_identico(self):
        for chave in TAX["habilidades"]:
            if chave == "9º|H17":
                continue
            ano, hab = chave.split("|")
            for s in dv.planejar_lote(ano, hab, 5, "Fácil", 1, taxonomia=TAX):
                self.assertNotIn("Eixo:", dv.sufixo_prompt(s), chave)


# --------------------------------------------------------------------------
# (d) dados insuficientes
# --------------------------------------------------------------------------
# Enunciados reais de 2026-10-01 (outputs/testes_locais/teste_20261001_*.json)
ONIBUS = ("Um ônibus transporta passageiros e, no trajeto, passa por quatro pontos formando um quadrilátero. "
          "Qual das alternativas mostra corretamente a classificação quanto aos lados desse quadrilátero?")
GARRAFAS = ("Em uma campanha de reciclagem, três garrafas PET foram separadas e suas bases formaram um "
            "triângulo. Qual das alternativas representa corretamente o tipo de triângulo formado?")
CORRETAS = [  # as 5 aprovadas pela auditoria
    "Em uma festa junina, as barracas formam triângulos. A barraca A tem todos os ângulos agudos, a barraca B "
    "tem um ângulo obtuso e a barraca C tem um ângulo reto. Como os triângulos podem ser classificados quanto "
    "aos ângulos?",
    "Triângulo tem lados de 5 cm, 5 cm e 8 cm. Qual tipo de triângulo esse é?",
    "Na festa junina, os organizadores decoraram uma barraca triangular com as seguintes medidas: lado A = 8 m, "
    "lado B = 10 m e lado C = 12 m. Como os lados são diferentes entre si, qual é o tipo de triângulo formado?",
    "Na biblioteca da escola, há um conjunto de prateleias formadas em triângulos. Um desses triângulos tem "
    "todos os lados iguais. Como se chama esse tipo de triângulo?",
    "Triângulo ABC tem lados AB=5 cm, BC=5 cm e CA=6 cm. Qual é o tipo de triângulo?",
]
SLOT_TRI = {"subtema": "triangulo", "tipo_raciocinio": "classificacao_lados"}
SLOT_QUAD = {"subtema": "quadrilatero", "tipo_raciocinio": "classificacao_propriedades"}


class TestDadosInsuficientes(unittest.TestCase):
    def test_casos_reais_sem_nenhum_dado(self):
        self.assertTrue(dv.dados_insuficientes_classificacao(q(ONIBUS), SLOT_QUAD))
        self.assertTrue(dv.dados_insuficientes_classificacao(q(GARRAFAS), SLOT_TRI))

    def test_conservadora_nao_acusa_as_corretas(self):
        for en in CORRETAS:
            self.assertFalse(dv.dados_insuficientes_classificacao(q(en), SLOT_TRI), en)

    def test_nao_acusa_itens_reais_do_banco(self):
        # No banco o dado costuma estar em texto_auxiliar; na geração tudo vai
        # para o enunciado, então os dois são juntados.
        con = sqlite3.connect(ROOT / "DB" / "questoes.db")
        rows = con.execute("SELECT enunciado_item, texto_auxiliar FROM itens WHERE ano='9º' "
                           "AND habilidade='H17' AND disciplina='Matemática'").fetchall()
        self.assertGreaterEqual(len(rows), 7)
        for en, aux in rows:
            self.assertFalse(dv.dados_insuficientes_classificacao(q(f"{aux or ''} {en or ''}"), SLOT_TRI), en)

    def test_alternativas_nao_mascaram_a_falta(self):
        cand = q(GARRAFAS, ("Equilátero", "Obtusângulo", "Retângulo", "Isósceles",
                            "Nenhuma das alternativas anteriores"))
        self.assertTrue(dv.dados_insuficientes_classificacao(cand, SLOT_TRI))

    def test_so_vale_para_slot_de_classificacao(self):
        self.assertFalse(dv.dados_insuficientes_classificacao(q(GARRAFAS), {"tipo_raciocinio": "identificacao"}))
        self.assertFalse(dv.dados_insuficientes_classificacao(q(GARRAFAS), None))

    def test_violacao_opt_in_e_restricao(self):
        slot = next(s for s in dv.planejar_lote("9º", "H17", 4, "Fácil", 0, taxonomia=TAX)
                    if s["subtema"] == "triangulo")
        sem = dv.violacoes_diversidade([], q(GARRAFAS), slot, "9º", "H17", 4, taxonomia=TAX)
        self.assertNotIn("dados_insuficientes", [v["tipo"] for v in sem])
        com = dv.violacoes_diversidade([], q(GARRAFAS), slot, "9º", "H17", 4, taxonomia=TAX,
                                       checar_dados_classificacao=True)
        self.assertIn("dados_insuficientes", [v["tipo"] for v in com])
        txt = dv.montar_restricao(com, "H17", slot)
        self.assertIn("medida", txt)
        self.assertIn("Eixo:", txt)


# --------------------------------------------------------------------------
# (e) SEM_CONTEXTO planejado
# --------------------------------------------------------------------------
class TestSemContexto(unittest.TestCase):
    def test_desligado_por_padrao(self):
        # Medido em 2026-10-01: com o modelo descartando o contexto em 5/20
        # slots, o SEM_CONTEXTO planejado colide e o context_diversity de P01
        # cai (0,890 -> 0,854 no simulador). G11 reprova qualquer queda.
        self.assertFalse(bt.SEM_CONTEXTO_EM_CONCEITUAIS)
        self.assertFalse([k for k, h in TAX["habilidades"].items() if "contextos" in h])
        for _, p in planos(TAX, "9º|H17", seeds=range(5)):
            self.assertFalse([s for s in p if s["contexto"] == dv.SEM_CONTEXTO])

    def test_ligado_so_nas_conceituais(self):
        com = {k for k, h in TAX_SEMCTX["habilidades"].items() if h.get("contextos")}
        self.assertEqual(com, CONCEITUAIS)
        for k in CONCEITUAIS:
            self.assertEqual(TAX_SEMCTX["habilidades"][k]["contextos"][0], dv.SEM_CONTEXTO)

    def test_rotulo_igual_ao_do_treino(self):
        self.assertEqual(dv.ROTULO_SEM_CONTEXTO, "sem contexto narrativo")
        self.assertEqual(dv.rotulo_contexto(dv.SEM_CONTEXTO), "sem contexto narrativo")
        p = dv.planejar_lote("9º", "H17", 17, "Fácil", 0, taxonomia=TAX_SEMCTX)
        s = next(x for x in p if x["contexto"] == dv.SEM_CONTEXTO)
        self.assertEqual(s["contexto_rotulo"], "sem contexto narrativo")
        self.assertIn(" Contexto: sem contexto narrativo.", dv.sufixo_prompt(s))

    def test_override_de_contextos(self):
        pool = ["sem_contexto", "construcao", "horta", "festa", "parque", "escola", "esporte", "feira"]
        tax = tax_com_overrides({"9º|H17": {"contextos": pool}})
        self.assertEqual(tax["habilidades"]["9º|H17"]["contextos"], pool)
        for _, p in planos(tax, "9º|H17", seeds=range(4)):
            self.assertTrue({s["contexto"] for s in p} <= set(pool))
        with self.assertRaises(ValueError):
            tax_com_overrides({"9º|H17": {"contextos": ["inexistente"]}})

    def test_sugestao_de_contexto_fica_no_pool_da_habilidade(self):
        # Antes vinha do pool global (podia sugerir "receita de bolo" a H17).
        pool = ["sem_contexto", "horta", "festa"]
        tax = tax_com_overrides({"9º|H17": {"contextos": pool}})
        slot = dict(dv.planejar_lote("9º", "H17", 3, "Fácil", 0, taxonomia=tax)[1], contexto="horta")
        aceitas = [q("Na horta comunitária, um canteiro triangular tem lados de 3 m, 3 m e 3 m. "
                     "Como ele é classificado quanto aos lados?")]
        cand = q("Na horta, um canteiro tem dois pares de lados paralelos e quatro ângulos retos, com lados "
                 "de 2 m e 5 m. Qual é o nome desse quadrilátero?")
        v = [x for x in dv.violacoes_diversidade(aceitas, cand, slot, "9º", "H17", 3, taxonomia=tax)
             if x["tipo"] == "contexto_repetido"]
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["sugerido"], "festa")


# --------------------------------------------------------------------------
# Correções da revisão adversarial de 2026-10-01
# --------------------------------------------------------------------------
class TestCorrecoesRevisao(unittest.TestCase):
    def test_medida_por_extenso_nao_e_falta_de_dado(self):
        # "noventa graus" É dado; a guarda acusava e o slot era regenerado à toa.
        for en in ("Um triângulo tem um ângulo de noventa graus. Classifique-o quanto aos ângulos.",
                   "Uma bandeirinha tem um ângulo de cento e vinte graus. Classifique-a.",
                   "Uma placa tem lados de cinco, cinco e cinco centímetros. Classifique-a.",
                   "Os ângulos de uma placa medem trinta graus, sessenta graus e noventa graus."):
            with self.subTest(en):
                self.assertFalse(dv.dados_insuficientes_classificacao(q(en), SLOT_TRI))
        # e continua acusando quando não há dado nenhum
        self.assertTrue(dv.dados_insuficientes_classificacao(q(GARRAFAS), SLOT_TRI))

    def test_palavras_chave_sem_lookbehind_e_classificacao_igual(self):
        """'(?<!triangulo )retangul' (lookbehind) virou máscara no texto: a
        classificação de subtema e de objeto não muda."""
        for k, h in TAX["habilidades"].items():
            for s in h["subtemas"]:
                pk = s["palavras_chave"] if isinstance(s["palavras_chave"], list) else [s["palavras_chave"]]
                for p in pk:
                    self.assertNotIn("(?<", p, (k, s["id"]))
        for _oid, rx in dv.OBJETOS:
            self.assertNotIn("(?<", rx)
        casos = (
            ("Um triângulo retângulo tem um ângulo de 90°. Quanto aos ângulos, ele é:",
             "triangulo", "triangulo"),
            ("Um retângulo tem lados de 3 cm e 5 cm. Qual é o nome?", "quadrilatero", "quadrilatero"),
            ("Um triângulo retângulo e um retângulo foram desenhados. Qual é o retângulo?",
             "quadrilatero", "triangulo"),
        )
        for en, subtema, objeto in casos:
            with self.subTest(en):
                c = dv.classificar_questao(q(en), "9º", "H17", TAX)
                self.assertEqual((c["subtema"], c["objeto_matematico"]), (subtema, objeto))
        # habilidade com subtema retangulo_quadrado: "triângulo retângulo" não conta
        chave = next(k for k, h in TAX["habilidades"].items()
                     if any(s["id"] == "retangulo_quadrado" for s in h["subtemas"])
                     and any(s["id"] == "triangulo" for s in h["subtemas"]))
        ano, hab = chave.split("|")
        c = dv.classificar_questao(q("Um triângulo retângulo isósceles foi recortado."), ano, hab, TAX)
        self.assertEqual(c["subtema"], "triangulo")


if __name__ == "__main__":
    unittest.main()
