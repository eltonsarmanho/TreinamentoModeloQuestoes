"""Testes da curadoria de diversidade (unittest; roda também com pytest)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import curar_diversidade as cd  # noqa: E402

USER = "Gere 1 questão(ões) de matemática. Ano: 9º ano. Habilidade: H17 — Classificar. Dificuldade: Fácil."


def ex(enunciado, meta=None, user=USER):
    q = {"enunciado": enunciado, "alternativas": {k: k for k in "ABCDE"},
         "resolucao_passo_a_passo": "x", "resposta_correta": "A", "difficulty": "EASY"}
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
            "meta": meta if meta is not None else {"ano": "9º", "habilidade": "H17"}}


TRI = "Um triângulo tem lados de {} cm, {} cm e {} cm. Como ele é classificado quanto aos lados?"


class TestCuradoria(unittest.TestCase):
    def setUp(self):
        self.base = [ex(TRI.format(3, 4, 5)),                      # real
                     ex("Um quadrilátero tem quatro lados iguais e quatro ângulos retos. Qual é o nome?")]
        self.sint = [ex(TRI.format(i, i + 1, i + 2), {"ano": "9º", "habilidade": "H17", "sintetico": True})
                     for i in range(6, 12)]

    def test_ano_hab_do_prompt_quando_meta_falta(self):
        self.assertEqual(cd.ano_habilidade(ex("x", meta={})), ("9º", "H17"))

    def test_mesmo_codigo_anos_diferentes_nao_mistura(self):
        outro = ex(TRI.format(3, 4, 5), {"ano": "5º", "habilidade": "H17"})
        rel, _ = cd.curar(self.base + [outro])
        chaves = {(g["ano"], g["habilidade"]) for g in rel["grupos"]}
        self.assertEqual(chaves, {("9º", "H17"), ("5º", "H17")})

    def test_so_sintetico_redundante_e_removido_real_nunca(self):
        reais_dup = [ex(TRI.format(3, 4, 5)), ex(TRI.format(7, 8, 9))]  # reais duplicados entre si
        dados = self.base + reais_dup + self.sint
        rel, _ = cd.curar(dados)
        rem = set(rel["remover_ex_idx"])
        self.assertEqual(rem, set(range(4, 10)))  # todos os sintéticos (troca de números)
        for i in rem:
            self.assertEqual(cd.origem(dados[i]["meta"]), "sintetico")

    def test_concentracao_sinalizada(self):
        rel, _ = cd.curar(self.base + self.sint)
        g = rel["grupos"][0]
        self.assertEqual(g["dominante"], "triangulo")
        self.assertTrue(g["concentracao_excessiva"])
        self.assertGreater(g["duplicate_rate"], 0.5)

    def test_plano_pede_subtema_sub_representado(self):
        rel, _ = cd.curar(self.base + self.sint)
        plano = cd.plano_destilacao(rel)
        subs = {(i["ano"], i["habilidade"], i["subtema"]): i["quantidade"] for i in plano["itens"]}
        self.assertIn(("9º", "H17", "quadrilatero"), subs)
        self.assertGreater(subs[("9º", "H17", "quadrilatero")], 0)
        self.assertEqual(plano["total"], sum(subs.values()))

    def test_apply_nao_duplica_e_anexa_sufixo(self):
        dados = self.base + self.sint
        rel, regs = cd.curar(dados)
        novos, n_suf = cd.aplicar(dados, rel, regs, sufixo=True)
        self.assertEqual(len(novos), len(dados) - rel["n_remover"])
        self.assertLessEqual(len(novos), len(dados))
        user = novos[0]["messages"][1]["content"]
        self.assertTrue(user.startswith(USER) and "Subtema: triângulos." in user)
        self.assertEqual(novos[0]["meta"]["subtema"], "triangulo")
        self.assertEqual(n_suf, 2)
        # schema da resposta intacto e entrada não mutada
        self.assertEqual(novos[0]["messages"][2], dados[0]["messages"][2])
        self.assertEqual(dados[0]["messages"][1]["content"], USER)

    def test_determinismo(self):
        a, _ = cd.curar(self.base + self.sint)
        b, _ = cd.curar(self.base + self.sint)
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_main_recusa_sobrescrever_train_e_val(self):
        with tempfile.TemporaryDirectory() as d:
            tr = Path(d) / "train.jsonl"
            tr.write_text("\n".join(json.dumps(e) for e in self.base + self.sint), encoding="utf-8")
            for alvo in (tr, Path(d) / "val.jsonl"):
                with self.assertRaises(SystemExit):
                    cd.main(["--train", str(tr), "--apply", "--saida", str(alvo),
                             "--plano", f"{d}/p.json", "--relatorio", f"{d}/r.json"])
            antes = tr.read_text(encoding="utf-8")
            cd.main(["--train", str(tr), "--apply", "--sufixo", "--saida", f"{d}/cur.jsonl",
                     "--plano", f"{d}/p.json", "--relatorio", f"{d}/r.json"])
            self.assertEqual(tr.read_text(encoding="utf-8"), antes)
            self.assertTrue((Path(d) / "r.md").exists())
            self.assertIn("itens", json.loads((Path(d) / "p.json").read_text()))


if __name__ == "__main__":
    unittest.main()


class TestGruposAusentes(unittest.TestCase):
    def test_plano_inclui_habilidade_sem_exemplo_no_treino(self):
        rel, _ = cd.curar([], incluir_ausentes=True)
        chaves = {(g["ano"], g["habilidade"]) for g in rel["grupos"]}
        self.assertIn(("5º", "H21"), chaves)
        self.assertGreater(rel["n_pedidos_professor"], 0)

    def test_padrao_nao_inclui(self):
        rel, _ = cd.curar([])
        self.assertEqual(rel["grupos"], [])
