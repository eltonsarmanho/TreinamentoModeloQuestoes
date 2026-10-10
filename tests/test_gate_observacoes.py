"""OBSERVAÇÕES informativas do gate (P0-4; src/observacoes_gate.py).

Garantias testadas:
  * relatório sem texto -> "n/d" sem erro (os históricos não têm `obj`);
  * cobertura de verificação, variabilidade entre rodadas e vícios de texto
    calculam o que dizem;
  * o veredito PROMOVIDO / NÃO PROMOVIDO e os gates NÃO mudam com as
    observações (nem quando elas quebram, nem com/sem texto), nos perfis
    multiseed e n1.
"""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import observacoes_gate as og  # noqa: E402
import promover_checkpoint as pc  # noqa: E402
from test_gate_multiseed import _rel  # noqa: E402


def _q(enunciado, e="Nenhum dos valores acima", gab="A"):
    return {"enunciado": enunciado,
            "alternativas": {"A": "1", "B": "2", "C": "3", "D": "4", "E": e},
            "resolucao_passo_a_passo": "1 + 1 = 2", "resposta_correta": gab,
            "difficulty": "EASY"}


def _com_texto(rel, fabrica):
    """Cópia do relatório com `obj` em cada item: fabrica(indice, rodada) -> questão."""
    r = copy.deepcopy(rel)
    for i, d in enumerate(r["detalhes"]):
        d["obj"] = fabrica(i, r["seed_rodada"])
    return r


DISTINTA = [
    "Maria comprou {n} canetas azuis na papelaria perto da escola e pagou com uma nota",
    "Um tanque de água armazena {n} litros e esvazia lentamente durante a madrugada fria",
    "Pedro percorreu {n} quilômetros de bicicleta no parque municipal durante o fim de semana",
]


class TestObservacoesSemTexto(unittest.TestCase):
    def test_relatorios_antigos_dao_nd_sem_erro(self):
        bases = [_rel(r, "b") for r in range(3)]
        cands = [_rel(r, "c") for r in range(3)]
        obs = og.observacoes(bases, cands)
        for lado in ("baseline", "candidato"):
            self.assertEqual(obs[lado]["variabilidade_entre_rodadas"], "n/d")
            self.assertEqual(obs[lado]["vicios_de_texto"], "n/d")
        linhas = og.formatar(obs)
        self.assertTrue(all("OBSERVAÇÃO (não bloqueante)" in l for l in linhas))
        self.assertIn("n/d", "\n".join(linhas))

    def test_cobertura_vem_de_pos_processamento(self):
        r = _rel(0, "b")
        r["pos_processamento"].update({"ok": 12, "nao_verificavel": 15, "corrigido": 1,
                                       "depende_de_visual": 0})
        c = og.cobertura_verificacao([r])
        self.assertEqual(c["agregado"]["ok_pct"], 40.0)
        self.assertEqual(c["agregado"]["nao_verificavel_pct"], 50.0)
        self.assertEqual(c["por_rodada"][0]["rodada"], 0)

    def test_cobertura_agrega_rodadas(self):
        a, b = _rel(0, "b"), _rel(1, "b")
        a["pos_processamento"].update({"ok": 30, "nao_verificavel": 0})
        b["pos_processamento"].update({"ok": 0, "nao_verificavel": 30})
        c = og.cobertura_verificacao([a, b])
        self.assertEqual((c["agregado"]["ok_pct"], c["agregado"]["nao_verificavel_pct"]), (50.0, 50.0))
        self.assertEqual(len(c["por_rodada"]), 2)

    def test_uma_rodada_so_nao_tem_variabilidade(self):
        r = _com_texto(_rel(0, "b"), lambda i, rod: _q("Quanto é 2 + 3?"))
        self.assertEqual(og.variabilidade_entre_rodadas([r]), "n/d")


class TestObservacoesComTexto(unittest.TestCase):
    def test_variabilidade_repeticao_total_x_questoes_distintas(self):
        iguais = [_com_texto(_rel(r, "b"), lambda i, rod: _q(DISTINTA[0].format(n=10 + i)))
                  for r in range(3)]
        v = og.variabilidade_entre_rodadas(iguais)
        self.assertEqual(v["near_dup_pct"], 100.0)
        self.assertEqual(v["jaccard_medio"], 1.0)
        self.assertEqual(v["itens_comparados"], 30)
        self.assertEqual(v["pares"], 30 * 3)  # C(3,2) pares por item

        distintos = [_com_texto(_rel(r, "b"),
                                lambda i, rod: _q(DISTINTA[rod % 3].format(n=i)))
                     for r in range(3)]
        v = og.variabilidade_entre_rodadas(distintos)
        self.assertLess(v["jaccard_medio"], 0.2)
        self.assertEqual(v["near_dup_pct"], 0.0)

    def test_sufixo_avisa_que_o_prompt_muda(self):
        rels = [_com_texto(_rel(r, "b"), lambda i, rod: _q("x y z w v u t")) for r in range(2)]
        for r in rels:
            r["inferencia"] = {"sufixo": True}
        v = og.variabilidade_entre_rodadas(rels)
        self.assertFalse(v["prompt_identico_entre_rodadas"])
        self.assertIn("nota", v)

    def test_vicios_nenhuma_das_anteriores_e_caixa_alta(self):
        def fab(i, rod):
            if i < 6:   # E = nenhuma das alternativas anteriores (2 delas são o gabarito)
                return _q("Quanto vale a soma abaixo hoje?", e="Nenhuma das alternativas anteriores",
                          gab="E" if i < 2 else "A")
            if i < 9:   # CAIXA ALTA
                return _q("VEJA OS BLOCOS DE MONTAR QUE PEDRO POSSUI NA CAIXA", e="8")
            return _q("Quanto vale a soma abaixo hoje?", e="Nenhuma pessoa chegou")  # não conta
        t = og.vicios_de_texto([_com_texto(_rel(0, "b"), fab)])
        self.assertEqual(t["questoes_com_texto"], 30)
        self.assertEqual((t["nenhuma_das_anteriores_n"], t["nenhuma_e_o_gabarito_n"]), (6, 2))
        self.assertEqual((t["caixa_alta_n"], t["caixa_alta_pct"]), (3, 10.0))
        self.assertEqual(t["nenhuma_das_anteriores_pct"], 20.0)

    def test_detectores_individuais(self):
        for texto in ("Nenhuma das anteriores", "NENHUMA DAS ALTERNATIVAS ACIMA",
                      "Nenhuma das opções anteriores", "Nenhum dos itens acima"):
            self.assertTrue(og.e_nenhuma_das_anteriores(_q("x", e=texto)), texto)
        for texto in ("Nenhuma pessoa", "8", "Nenhum aluno faltou"):
            self.assertFalse(og.e_nenhuma_das_anteriores(_q("x", e=texto)), texto)
        self.assertFalse(og.e_caixa_alta(_q("R$ 5 + 3")))          # curto demais
        self.assertFalse(og.e_caixa_alta(_q("Pedro tem 3 BALAS e 2 doces para repartir")))
        self.assertTrue(og.e_caixa_alta(_q("PEDRO TEM 3 BALAS E 2 DOCES PARA REPARTIR")))

    def test_texto_do_avaliar_diversidade_tambem_serve_para_vicios(self):
        rel = {"modo_atual": "ajustado", "modos": {"ajustado": {"lotes": [{
            "id": "P01", "questoes": [_q("Quanto vale a soma abaixo hoje?",
                                         e="Nenhuma das anteriores")]}]}}}
        self.assertEqual(og.vicios_de_texto([rel])["nenhuma_das_anteriores_n"], 1)


class TestVereditoNaoMuda(unittest.TestCase):
    def _args(self, d, bases, cands, saida):
        ba, ca = [], []
        for tag, rels, dest in (("b", bases, ba), ("c", cands, ca)):
            for i, r in enumerate(rels):
                p = Path(d) / f"{tag}{i}.json"
                p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                dest.append(str(p))
        return Namespace(baseline_gguf_seeds=ba, candidato_gguf_seeds=ca,
                         comparar_inferencia=False, saida=str(saida))

    def _roda(self, bases, cands, patch_obs=None):
        with tempfile.TemporaryDirectory() as d:
            saida = Path(d) / "v.json"
            args = self._args(d, bases, cands, saida)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if patch_obs:
                    with mock.patch.object(og, "observacoes", side_effect=patch_obs):
                        rc = pc.main_multiseed(args, lambda p: json.loads(Path(p).read_text()))
                else:
                    rc = pc.main_multiseed(args, lambda p: json.loads(Path(p).read_text()))
            return rc, json.loads(saida.read_text(encoding="utf-8")), buf.getvalue()

    @staticmethod
    def _nucleo(v):
        return (v["decisao"], v["motivo"], v["ganhos"],
                [(g["id"], g["bloqueante"], g["passou"], g["detalhe"]) for g in v["gates"]])

    def _cenarios(self):
        bases = [_rel(r, "b") for r in range(3)]
        melhor = [_rel(r, "c") for r in range(3)]
        for r in melhor:   # candidato com ganho: mais verificáveis? -> menos regenerações
            r["pos_processamento"]["regeneracoes_total"] = 1
        pior = [_rel(r, "c", falhas=range(0, 6)) for r in range(3)]  # G2* reprova
        return bases, melhor, pior

    def test_promovido_e_nao_promovido_identicos_com_e_sem_observacoes(self):
        bases, melhor, pior = self._cenarios()
        vistos = set()
        for cands in (melhor, pior):
            rc0, v0, _ = self._roda(bases, cands)
            vistos.add(v0["decisao"])
            # (a) a observação levanta: calcular_seguro engole, veredito igual
            rc1, v1, out1 = self._roda(bases, cands, patch_obs=RuntimeError("boom"))
            self.assertEqual(rc0, rc1)
            self.assertEqual(self._nucleo(v0), self._nucleo(v1))
            self.assertIn("erro", v1["observacoes"])
            self.assertIn("não calculada", out1)
            # (b) relatórios COM texto: mesmo veredito
            com = lambda rels: [_com_texto(r, lambda i, rod: _q("Quanto é 2 + 3 agora?"))
                                for r in rels]
            rc2, v2, _ = self._roda(com(bases), com(cands))
            self.assertEqual(rc0, rc2)
            self.assertEqual(self._nucleo(v0), self._nucleo(v2))
            self.assertNotEqual(v2["observacoes"]["candidato"]["vicios_de_texto"], "n/d")
        self.assertEqual(vistos, {"PROMOVIDO", "NÃO PROMOVIDO"})

    def test_observacoes_impressas_apos_a_decisao_e_gravadas(self):
        bases, melhor, _ = self._cenarios()
        rc, v, out = self._roda(bases, melhor)
        self.assertEqual(rc, 0)
        ult = out.rindex("DECISÃO:")
        self.assertLess(ult, out.index("OBSERVAÇÃO (não bloqueante)"))
        self.assertEqual(set(v["observacoes"]), {"nota", "baseline", "candidato"})
        # nenhuma observação virou gate
        self.assertFalse(any("OBSERV" in g["nome"].upper() for g in v["gates"]))

    def test_perfil_n1_grava_observacoes_sem_mudar_veredito(self):
        base, cand = _rel(0, "b"), _rel(0, "c")
        cand["pos_processamento"]["regeneracoes_total"] = 1
        with tempfile.TemporaryDirectory() as d:
            pb, pc_, ps = (Path(d) / n for n in ("b.json", "c.json", "v.json"))
            pb.write_text(json.dumps(base)), pc_.write_text(json.dumps(cand))
            argv = ["promover_checkpoint.py", "--baseline-gguf", str(pb),
                    "--candidato-gguf", str(pc_), "--saida", str(ps)]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                rc = pc.main()
            v = json.loads(ps.read_text(encoding="utf-8"))
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(og, "observacoes", side_effect=ValueError("x")), \
                    contextlib.redirect_stdout(io.StringIO()):
                rc2 = pc.main()
            v2 = json.loads(ps.read_text(encoding="utf-8"))
        self.assertEqual(rc, rc2)
        self.assertEqual(self._nucleo(v), self._nucleo(v2))
        self.assertEqual(v["observacoes"]["baseline"]["variabilidade_entre_rodadas"], "n/d")
        self.assertIn("erro", v2["observacoes"])


if __name__ == "__main__":
    unittest.main()
