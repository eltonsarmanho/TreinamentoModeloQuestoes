"""Testes da calibração dos agentes (src/calibrar_agentes.py).

Sem rede e sem custo: cliente SIMULADO e arquivos em diretório temporário.
Cobrem o que protege o orçamento (teto global persistente entre execuções,
cache que torna a repetição gratuita), a seleção dos itens reais (exclusão de
dependência visual) e a contagem da matriz de confusão.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import agentes_questoes as aq  # noqa: E402
import calibrar_agentes as ca  # noqa: E402


def _args(tmp, *extra):
    return ca.construir_parser().parse_args(["--dry-run", "--dir-dry-run", str(tmp), *extra])


def _linha(boa, val, rev, conjunto="cal20", id_="x"):
    j = lambda v: {"veredito": v, "avaliado": True, "problemas": [], "criterios_falhos": []}  # noqa: E731
    return {"id": id_, "conjunto": conjunto, "boa": boa, "validador": j(val), "revisor": j(rev),
            "par": val and rev}


class FakeRow(dict):
    def keys(self):
        return list(super().keys())


def _row(i, enunciado, alts=("1", "2", "3", "4"), imagem=None, justificativa="Conta: 1 + 1 = 2, logo B.",
         ano="5º", hab="H01"):
    r = {"disciplina": "Matemática", "imagem": imagem, "enunciado_item": enunciado, "texto_auxiliar": "",
         "gabarito": "B", "grau_resolucao": "Fácil", "codigo_item": f"COD{i:03d}", "ano": ano,
         "habilidade": hab, "descricao_item": "Resolver problemas", "lote": None, "justificativa_geral": "",
         "depende_de_imagem": 0, "descricao_imagem": None, "imagem_path": None}
    for k, L in enumerate("abcd"):
        r[f"alternativa_{L}"] = alts[k]
        r[f"alternativa_{L}_imagem"] = None
        r[f"justificativa_alternativa_{L}"] = justificativa if L == "b" else ""
    return FakeRow(r)


class TestMatriz(unittest.TestCase):
    def test_contagem_e_par(self):
        linhas = [_linha(True, True, True, id_="b1"), _linha(True, True, False, id_="b2"),
                  _linha(False, True, True, id_="r1"), _linha(False, False, True, id_="r2"),
                  _linha(False, False, False, id_="r3")]
        m = ca.metricas(linhas, [])
        self.assertEqual({k: m["cal20"]["par"][k] for k in ("vp", "fn", "fp", "vn")},
                         {"vp": 1, "fn": 1, "fp": 1, "vn": 2})
        self.assertEqual(m["cal20"]["validador"]["fp"], 1)
        self.assertEqual(m["cal20"]["revisor"]["fp"], 2)
        self.assertEqual(m["cal20"]["ruins_aceitas_pelo_par"], ["r1"])

    def test_falsa_rejeicao_e_gatilho_de_ajuste(self):
        cal = [_linha(False, False, False, id_="r1"), _linha(True, True, True, id_="b1")]
        reais = [_linha(True, True, True, "real30", f"q{i}") for i in range(8)]
        reais += [_linha(True, False, True, "real30", "q8"), _linha(True, True, False, "real30", "q9")]
        m = ca.metricas(cal, reais)
        self.assertEqual(m["real30"]["par"]["rejeitados"], 2)
        self.assertAlmostEqual(m["real30"]["par"]["taxa_falsa_rejeicao"], 0.2)
        self.assertFalse(m["precisa_ajuste"])  # 20% não é "mais de 20%"
        reais.append(_linha(True, False, False, "real30", "q10"))
        self.assertTrue(ca.metricas(cal, reais)["precisa_ajuste"])


class TestSelecaoReal(unittest.TestCase):
    def test_exclui_visual_e_sem_resolucao(self):
        rows = [_row(1, "Quanto é 1 + 1?"),
                _row(2, "Observe a figura abaixo. Quanto mede o lado?"),
                _row(3, "Quanto é 1 + 1?", imagem=b"png"),
                _row(4, "Quanto é 1 + 1?", justificativa="B"),
                _row(5, "Um triângulo tem lados 3, 4 e 5. Quanto aos lados ele é:")]
        itens, info = ca.carregar_real(n=5, n_geo=1, rows=rows)
        ids = {it["id"] for it in itens}
        self.assertEqual(ids, {"REAL-COD001", "REAL-COD005"})
        self.assertEqual(info["excluidos"].get("imagem"), 1)
        self.assertEqual(info["excluidos"].get("sem_resolucao"), 1)
        self.assertEqual(info["excluidos"].get("remissao_visual"), 1)
        # o E sintético do treino está presente (mesmo conversor)
        self.assertEqual(itens[0]["questao"]["alternativas"]["E"], "Nenhuma das alternativas anteriores")

    def test_amostra_deterministica(self):
        rows = [_row(i, f"Quanto é {i} + 1?", ano=a) for i, a in enumerate(["2º", "5º", "9º"] * 6)]
        a, _ = ca.carregar_real(n=6, n_geo=0, rows=rows)
        b, _ = ca.carregar_real(n=6, n_geo=0, rows=rows)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
        self.assertEqual(sorted(x["ano"] for x in a), ["2º", "2º", "5º", "5º", "9º", "9º"])


class TestOrcamentoECache(unittest.TestCase):
    def test_teto_global_persistente(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "ledger.json").write_text(json.dumps({"chamadas": 158}), encoding="utf-8")
            args = _args(tmp, "--max-chamadas", "50", "--teto-global", "160")
            ag = ca.montar(args, "r", ca.caminhos(args))
            self.assertEqual(ag.orcamento.max_chamadas, 2)
            q = ca.carregar_cal20()[0]["questao"]
            ag.chamar("validador", "fase1", [{"role": "system", "content": aq.VALIDADOR_SISTEMA},
                                             {"role": "user", "content": "x"}])
            ag.chamar("revisor", "revisao", [{"role": "system", "content": aq.REVISOR_SISTEMA},
                                             {"role": "user", "content": "y"}])
            with self.assertRaises(aq.OrcamentoEsgotado):
                ag.validar(q)
            self.assertEqual(json.loads((tmp / "ledger.json").read_text())["chamadas"], 160)

    def test_repeticao_custa_zero(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            args = _args(tmp, "--rodada", "t", "--conjunto", "cal", "--ids", "R1-Q1,R1-Q3")
            ca.executar(args)
            gasto1 = json.loads((tmp / "ledger.json").read_text())["chamadas"]
            self.assertGreater(gasto1, 0)
            saida = ca.executar(args)
            self.assertEqual(json.loads((tmp / "ledger.json").read_text())["chamadas"], gasto1)
            ex = saida["rodadas"]["t"]["execucoes"][-1]
            self.assertEqual(ex["chamadas_reais"], 0)
            self.assertGreater(ex["cache_hits"], 0)
            self.assertEqual(set(saida["rodadas"]["t"]["itens"]), {"R1-Q1", "R1-Q3"})

    def test_real_exige_max_chamadas(self):
        args = ca.construir_parser().parse_args(["--conjunto", "cal", "--ids", "R1-Q1",
                                                 "--saida", "/tmp/claude-1000/base/x.json"])
        with self.assertRaises(SystemExit):
            ca.executar(args)


class TestConjuntosR5(unittest.TestCase):
    """Recalibração de 2026-10-01: defeitos conhecidos, D5 e metas novas."""

    def test_defeitos_confere_codigo(self):
        itens = ca.carregar_defeitos()
        self.assertEqual({it["id"] for it in itens}, {"DEF-1368", "DEF-1080", "DEF-1052", "DEF-561", "DEF-881"})
        self.assertEqual([it["id"] for it in itens if it["boa"]], ["DEF-881"])
        with self.assertRaises(ValueError):
            ca.carregar_defeitos(casos=[(1368, "OUTRO-CODIGO", False, "x")])

    def test_d5_copia_literal_e_controles(self):
        itens = {it["id"]: it for it in ca.carregar_d5()}
        self.assertEqual(sum(not it["boa"] for it in itens.values()), 6)
        self.assertEqual(sum(it["boa"] for it in itens.values()), 6)
        b1 = itens["D5-B1"]["questao"]
        self.assertEqual(b1["alternativas"]["D"], "Acutângulo")
        self.assertEqual(itens["D5-B2"]["questao"]["alternativas"]["E"], "Escaleno")
        for it in itens.values():
            q = it["questao"]
            self.assertIn(q["resposta_correta"], "ABCDE")
            self.assertTrue(it["descricao"], it["id"])

    def test_metas_r5(self):
        linhas = [_linha(False, False, False, "defeitos", "d1"), _linha(True, True, True, "defeitos", "ok"),
                  _linha(False, True, True, "d5", "b1"), _linha(True, True, True, "d5", "g1")]
        reais = [_linha(True, True, True, "real30", f"q{i}") for i in range(6)]
        reais.append(_linha(True, False, False, "real30", "REAL-MT5023MH05MT"))
        m = ca.metricas_r5(linhas + reais)
        self.assertEqual(m["real_falsa_rejeicao"]["ajustada"], "0/6")
        self.assertEqual(m["d5"]["par"]["ruins_aceitas"], ["b1"])
        self.assertFalse(m["metas_ok"])
        self.assertEqual(m["geral"]["n"], 11)
        self.assertEqual(m["geral"]["reprova_par"]["n"], 2)

    def test_dry_run_todos(self):
        with tempfile.TemporaryDirectory() as d:
            args = _args(Path(d), "--rodada", "t", "--conjunto", "todos", "--ids", "DEF-881,D5-B1,R1-Q1")
            saida = ca.executar(args)
            self.assertEqual(set(saida["rodadas"]["t"]["itens"]), {"DEF-881", "D5-B1", "R1-Q1"})
            self.assertIn("r5", saida["rodadas"]["t"]["metricas"])


if __name__ == "__main__":
    unittest.main()
