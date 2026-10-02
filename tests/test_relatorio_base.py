"""Testes do relatório da base (unittest; roda também com pytest).

Cobrem só as regras que decidem números do relatório (origem, meta, déficit)
e um ponta a ponta em arquivos temporários — nunca leem/escrevem data/ real.
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import relatorio_base as rb  # noqa: E402


class TestRegras(unittest.TestCase):
    def test_origem(self):
        db = {"MT9001FH17MT", "EF01MA02-001-L2-2026-09"}
        self.assertEqual(rb.origem_exemplo({"codigo_item": "MT9001FH17MT"}, db), "real_banco")
        self.assertEqual(rb.origem_exemplo({"codigo_item": "EF01MA02-001-L2-2026-09"}, db), "real_banco")
        self.assertEqual(rb.origem_exemplo({"codigo_item": "DIST-H17-Fácil-00001"}, db), "destilado")
        self.assertEqual(rb.origem_exemplo({"codigo_item": "X", "destilado": True}, db), "destilado")
        self.assertEqual(rb.origem_exemplo({"codigo_item": "SINT-H06-Fácil-0001"}, db), "sintetico")
        # Código desconhecido sem flag NÃO é presumido real: vira alerta.
        self.assertEqual(rb.origem_exemplo({"codigo_item": "MT0000"}, db), "real_fora_do_banco")

    def test_meta(self):
        self.assertEqual(rb.calcular_meta(1), 30)
        self.assertEqual(rb.calcular_meta(9), 30)       # 27 < 30
        self.assertEqual(rb.calcular_meta(11), 33)      # 3*11
        self.assertEqual(rb.calcular_meta(0), 30)
        self.assertEqual(rb.calcular_meta(5, meta_min=31), 33)  # arredonda p/ múltiplo de 3

    def test_deficit_e_o_maior_dos_tres(self):
        # Total já cumprido, mas tudo em Fácil: o déficit vem da dificuldade.
        d = rb.calcular_deficit(30, 30, {"Fácil": 30}, {"a": 30}, ["a"])
        self.assertEqual(d["deficit"], 20)
        # Subtemas vazios dominam quando há muitos.
        d = rb.calcular_deficit(30, 28, {"Fácil": 10, "Moderado": 10, "Difícil": 8},
                                {"a": 28}, ["a", "b", "c"])
        self.assertEqual(d["falta_por_subtema"], {"b": 3, "c": 3})
        self.assertEqual(d["deficit"], 6)
        # Acima da meta e equilibrado: zero (nunca negativo).
        d = rb.calcular_deficit(30, 60, {"Fácil": 20, "Moderado": 20, "Difícil": 20}, {"a": 60}, ["a"])
        self.assertEqual(d["deficit"], 0)


def _exemplo(codigo, enunciado, dif="Fácil", resolucao="3 + 4 = 7", letra="A", extra=None):
    q = {"enunciado": enunciado, "alternativas": {"A": "7", "B": "8", "C": "9", "D": "10", "E": "11"},
         "resolucao_passo_a_passo": resolucao, "resposta_correta": letra,
         "difficulty": {"Fácil": "EASY", "Moderado": "MEDIUM", "Difícil": "HARD"}[dif]}
    meta = {"codigo_item": codigo, "ano": "9º", "habilidade": "H17", "dificuldade": dif, **(extra or {})}
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                         {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
            "meta": meta}


class TestPontaAPonta(unittest.TestCase):
    def test_relatorio_em_arquivos_temporarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db = tmp / "q.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE itens (codigo_item TEXT, ano TEXT, habilidade TEXT, descricao_item TEXT, "
                        "imagem BLOB, depende_de_imagem INTEGER, grau_resolucao TEXT, lote TEXT)")
            con.executemany("INSERT INTO itens VALUES (?,?,?,?,?,?,?,?)", [
                ("MT9001FH17MT", "9º", "H17", "Classificar triângulos.", None, 0, "Fácil", None),
                ("MT9002FH17MT", "9º", "H17", "Classificar triângulos.", None, 0, "Moderado", None),
                ("MT9003FH17MT", "9º", "H17", "Classificar triângulos.", b"png", 0, "Difícil", None),
            ])
            con.commit()
            con.close()
            train = tmp / "t.jsonl"
            tri = "Um triângulo tem lados de {} cm, {} cm e {} cm. Como ele é classificado quanto aos lados?"
            exs = [_exemplo("MT9001FH17MT", tri.format(3, 4, 5)),
                   _exemplo("DIST-H17-Fácil-00001", tri.format(5, 6, 7), extra={"destilado": True}),
                   _exemplo("SINT-H17-Fácil-0001", "Um quadrado tem 4 lados. Quanto é 3 + 4?",
                            resolucao="3 + 4 = 7", letra="B", extra={"sintetico": True})]
            train.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in exs), encoding="utf-8")
            rel = rb.gerar_relatorio(train, db, None, [], 30, 3)
            h = next(x for x in rel["habilidades"] if (x["ano"], x["habilidade"]) == ("9º", "H17"))
            self.assertEqual(h["origem"], {"real_banco": 1, "destilado": 1, "sintetico": 1})
            # O destilado é near-dup do real (só números trocados): o REAL fica.
            self.assertEqual(h["near_dup"]["n"], 1)
            self.assertEqual(h["near_dup"]["exemplos"][0]["igual_a"], "MT9001FH17MT")
            # Gabarito B com conta = 7 (alternativa A): inconsistente e não conta como efetiva.
            self.assertEqual(h["gabarito"]["inconsistente"], 1)
            self.assertEqual(h["n_efetivas"], 1)
            self.assertEqual(h["banco"], {"total": 3, "textuais": 2, "usados_no_treino": 1,
                                          "graus": ["Fácil", "Moderado", "Difícil"], "em_validacao": 0,
                                          "textuais_livres": 1, "codigos_livres": ["MT9002FH17MT"]})
            self.assertEqual(h["deficit"], 29)
            md = rb.gerar_md(rel)
            self.assertIn("Déficit total a injetar", md)


if __name__ == "__main__":
    unittest.main()
