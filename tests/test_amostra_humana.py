"""Amostra humana cega (src/amostra_humana.py): gerar e apurar.

ATENÇÃO: tudo aqui é FIXTURE SINTÉTICA. Os relatórios e as "avaliações do
professor" são gerados por script (valores determinísticos inventados só para
exercitar o código); NÃO representam nenhuma avaliação real de questão.
"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import amostra_humana as ah  # noqa: E402
import unidades_tematicas as ut  # noqa: E402

# (ano, habilidade) cobrindo as 5 unidades, 4 itens de cada = 20 itens por relatório
ITENS = [("5º", "H01"), ("5º", "H03"), ("9º", "H02"), ("9º", "H05"),            # Números
         ("5º", "H07"), ("9º", "H10"), ("9º", "H12"), ("2º", "H09"),             # Álgebra
         ("5º", "H12"), ("9º", "H14"), ("9º", "H17"), ("2º", "H11"),             # Geometria
         ("5º", "H15"), ("9º", "H20"), ("9º", "H21"), ("2º", "H14"),             # Grandezas
         ("5º", "H20"), ("9º", "H25"), ("9º", "H26"), ("2º", "H19")]             # Prob./estat.


def _questao(origem, rodada, i):
    # Texto distinto por (origem, rodada, item), sem mencionar a origem.
    objetos = ["lápis", "bolas", "cadernos", "maçãs", "livros", "figurinhas"]
    return {"enunciado": f"Item sintético {(ord(origem[0]) * 7 + rodada * 13 + i * 17) % 997}: Ana tem {10 + i * 3 + rodada} "
                         f"{objetos[(i + rodada) % 6]} e ganha {2 + i} {objetos[(i * 2 + rodada) % 6]}. "
                         f"Quantos objetos ela tem agora?",
            "alternativas": {l: f"{k + i + rodada}" for k, l in enumerate("ABCDE")},
            "resolucao_passo_a_passo": f"{10 + i} + {2 + i} = {12 + 2 * i}",
            "resposta_correta": "ABCDE"[i % 5], "difficulty": "EASY"}


def _relatorio(origem, rodada=0, com_texto=True, sha="sha-oculto"):
    det = []
    for i, (ano, hab) in enumerate(ITENS):
        d = {"codigo_item_ref": f"IT{i:02d}", "ano": ano, "habilidade": hab,
             "status": "ok" if i % 2 else "nao_verificavel", "geometria": "nao_aplicavel",
             "difficulty_pedida": "Fácil"}
        if com_texto:
            d["obj"] = _questao(origem, rodada, i)
        det.append(d)
    return {"artefato": f"outputs/{origem}.gguf", "artefato_sha256": f"{sha}-{origem}",
            "seed_rodada": rodada, "num_amostras": len(det), "detalhes": det,
            "inferencia": {"sufixo": origem == "candidato"}}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def relatorio(self, nome, **kw):
        p = self.d / f"{nome}.json"
        p.write_text(json.dumps(_relatorio(**kw), ensure_ascii=False), encoding="utf-8")
        return str(p)

    def gerar(self, specs, saida="out", **kw):
        argv = ["gerar", "--relatorios", *specs, "--saida-dir", str(self.d / saida)]
        for k, v in kw.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ah.main(argv), 0)
        return self.d / saida

    def specs(self):
        return [f"baseline={self.relatorio('b0', origem='baseline', rodada=0)}",
                f"baseline={self.relatorio('b1', origem='baseline', rodada=1)}",
                f"candidato={self.relatorio('c0', origem='candidato', rodada=0)}",
                f"candidato={self.relatorio('c1', origem='candidato', rodada=1)}"]

    @staticmethod
    def le(saida):
        with open(saida / "folha_professor.csv", encoding="utf-8-sig", newline="") as f:
            linhas = list(csv.DictReader(f))
        return linhas, json.loads((saida / "chave_oculta.json").read_text(encoding="utf-8"))


class TestGerar(Base):
    def test_amostra_estratificada_sem_repeticao_e_embaralhada(self):
        out = self.gerar(self.specs(), n=20, min_por_unidade=3, seed=7)
        linhas, chave = self.le(out)
        self.assertEqual(len(linhas), 20)
        ids = [l["id"] for l in linhas]
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(set(ids), set(chave["itens"]))
        # estratificação: todas as unidades presentes, igualadas (20 / 5 = 4)
        por_u = {}
        for i in ids:
            u = chave["itens"][i]["unidade"]
            por_u[u] = por_u.get(u, 0) + 1
        self.assertEqual(set(por_u), set(ut.UNIDADES))
        self.assertEqual(set(por_u.values()), {4})
        # nunca a mesma questão duas vezes
        self.assertEqual(len({(l["enunciado"], l["A"], l["B"]) for l in linhas}), 20)
        # as duas origens aparecem em cada unidade
        for u in ut.UNIDADES:
            orig = {chave["itens"][i]["origem"] for i in ids if chave["itens"][i]["unidade"] == u}
            self.assertEqual(orig, {"baseline", "candidato"}, u)
        # ordem embaralhada: não vem agrupada por unidade
        seq = [chave["itens"][i]["unidade"] for i in ids]
        blocos = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        self.assertGreater(blocos, 10)
        # cada linha da chave aponta para o item certo
        for l in linhas:
            k = chave["itens"][l["id"]]
            self.assertEqual((l["ano"], l["habilidade"]), (k["ano"], k["habilidade"]))
            self.assertEqual(ut.unidade_de(l["ano"], l["habilidade"]), k["unidade"])

    def test_minimo_por_unidade_com_n_pequeno(self):
        out = self.gerar(self.specs(), n=10, min_por_unidade=2, seed=1)
        linhas, chave = self.le(out)
        cont = {}
        for i in (l["id"] for l in linhas):
            cont[chave["itens"][i]["unidade"]] = cont.get(chave["itens"][i]["unidade"], 0) + 1
        self.assertEqual(sorted(cont.values()), [2] * 5)

    def test_minimo_incompativel_com_n_aborta(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.gerar(self.specs(), n=5, min_por_unidade=4)

    def test_deterministica_na_seed(self):
        s = self.specs()
        a = (self.gerar(s, "a", n=20, seed=3) / "folha_professor.csv").read_bytes()
        b = (self.gerar(s, "b", n=20, seed=3) / "folha_professor.csv").read_bytes()
        c = (self.gerar(s, "c", n=20, seed=4) / "folha_professor.csv").read_bytes()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_folha_sem_nenhum_indicio_de_origem(self):
        out = self.gerar(self.specs(), n=20, seed=5)
        bruto = (out / "folha_professor.csv").read_text(encoding="utf-8-sig").lower()
        linhas, _ = self.le(out)
        self.assertEqual(list(linhas[0]), ah.COLUNAS_FOLHA)
        for proibido in ("baseline", "candidato", "sha-oculto", "sufixo", "gguf", "nao_verificavel",
                         "geometria", "status", "rodada", "seed", "b0.json", "c1.json", "outputs/"):
            self.assertNotIn(proibido, bruto, proibido)
        # a rubrica sai em branco
        for l in linhas:
            self.assertEqual({l[k] for k in ah.OBRIGATORIAS} | {l["comentario"]}, {""})

    def test_mesma_questao_em_dois_relatorios_nao_repete(self):
        a = self.relatorio("a", origem="baseline", rodada=0)
        b = self.relatorio("b", origem="baseline", rodada=0)   # texto idêntico ao de 'a'
        out = self.gerar([f"baseline={a}", f"candidato={b}"], n=20, seed=2)
        linhas, _ = self.le(out)
        self.assertEqual(len(linhas), 20)
        self.assertEqual(len({l["enunciado"] for l in linhas}), 20)

    def test_relatorio_sem_texto_e_ignorado_e_todos_sem_texto_aborta(self):
        sem = self.relatorio("sem", origem="baseline", com_texto=False)
        com = self.relatorio("com", origem="candidato")
        out = self.gerar([f"baseline={sem}", f"candidato={com}"], n=8, min_por_unidade=1)
        _, chave = self.le(out)
        self.assertEqual({v["origem"] for v in chave["itens"].values()}, {"candidato"})
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.gerar([f"baseline={sem}"], "x")


def preenche_sintetico(linhas, chave, ruim_a_cada=4):
    """FIXTURE: 'professor' inventado por script. Candidato é marcado inválido
    a cada `ruim_a_cada` itens; baseline a cada 2. Não é avaliação real."""
    for k, l in enumerate(linhas):
        origem = chave["itens"][l["id"]]["origem"]
        invalida = (k % 2 == 0) if origem == "baseline" else (k % ruim_a_cada == 0)
        for d in ah.DIMENSOES:
            l[d] = "s"
        if invalida:
            l["correcao_matematica"] = "n"
        l["valida_geral"] = "n" if invalida else "s"
        l["comentario"] = "fixture sintética"


class TestApurar(Base):
    def _folha_preenchida(self, saida="out", **gen):
        out = self.gerar(self.specs(), saida, n=20, seed=11, **gen)
        linhas, chave = self.le(out)
        preenche_sintetico(linhas, chave)
        return out, linhas, chave

    def _escreve(self, linhas, nome="preenchida.csv"):
        p = self.d / nome
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ah.COLUNAS_FOLHA)
            w.writeheader()
            w.writerows(linhas)
        return p

    def _apura(self, folha, chave_path, *extra):
        saida = self.d / "apuracao.json"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = ah.main(["apurar", "--folha", str(folha), "--chave", str(chave_path),
                          "--saida", str(saida), *extra])
        res = json.loads(saida.read_text(encoding="utf-8")) if saida.exists() else None
        return rc, res, out.getvalue(), err.getvalue()

    def test_wilson_conhecido(self):
        lo, hi = ah.wilson(8, 10)
        self.assertAlmostEqual(lo, 49.02, places=1)
        self.assertAlmostEqual(hi, 94.34, places=1)

    def test_apuracao_confere_com_contagem_manual(self):
        out, linhas, chave = self._folha_preenchida()
        rc, res, texto, _ = self._apura(self._escreve(linhas), out / "chave_oculta.json")
        self.assertEqual(rc, 0)
        self.assertEqual(res["n_avaliadas"], 20)
        # contagem manual independente
        val = [l for l in linhas if l["valida_geral"] == "s"]
        self.assertEqual(res["geral"]["validade"]["s"], len(val))
        for u in ut.UNIDADES:
            do_u = [l for l in linhas if chave["itens"][l["id"]]["unidade"] == u]
            t = res["por_unidade"][u]["validade"]
            self.assertEqual((t["n"], t["s"]), (len(do_u), sum(l["valida_geral"] == "s" for l in do_u)))
            self.assertEqual(t["ic95_pct"], list(ah.wilson(t["s"], t["n"])))
        for o in ("baseline", "candidato"):
            do_o = [l for l in linhas if chave["itens"][l["id"]]["origem"] == o]
            t = res["por_origem"][o]["validade"]
            self.assertEqual((t["n"], t["s"]), (len(do_o), sum(l["valida_geral"] == "s" for l in do_o)))
        dim = res["geral"]["dimensoes"]["correcao_matematica"]
        self.assertEqual(dim["s"], sum(l["correcao_matematica"] == "s" for l in linhas))
        self.assertEqual(res["geral"]["dimensoes"]["resposta_unica"]["pct"], 100.0)
        self.assertIn("Por unidade temática", texto)
        self.assertIn("IC95", texto)

    def test_aceita_sim_nao_e_maiusculas(self):
        out, linhas, _ = self._folha_preenchida()
        for l in linhas:
            for k in ah.OBRIGATORIAS:
                l[k] = {"s": " Sim", "n": "NÃO"}[l[k]]
        rc, res, _, _ = self._apura(self._escreve(linhas), out / "chave_oculta.json")
        self.assertEqual(rc, 0)
        self.assertEqual(res["n_avaliadas"], 20)

    def test_folha_com_linha_vazia_e_valor_invalido_e_rejeitada(self):
        out, linhas, _ = self._folha_preenchida()
        for k in ah.OBRIGATORIAS:           # linha totalmente vazia
            linhas[0][k] = ""
        linhas[1]["valida_geral"] = "talvez"   # fora de s/n
        linhas[2]["contexto_plausivel"] = ""   # parcialmente em branco
        rc, res, _, err = self._apura(self._escreve(linhas), out / "chave_oculta.json")
        self.assertEqual(rc, 2)
        self.assertIsNone(res)
        self.assertIn("linha vazia", err)
        self.assertIn("valor fora de s/n", err)
        self.assertIn("'talvez'", err)
        self.assertIn("campos em branco: contexto_plausivel", err)

    def test_ignorar_incompletas_apura_so_as_completas(self):
        out, linhas, _ = self._folha_preenchida()
        for k in ah.OBRIGATORIAS:
            linhas[0][k] = ""
        rc, res, _, err = self._apura(self._escreve(linhas), out / "chave_oculta.json",
                                      "--ignorar-incompletas")
        self.assertEqual(rc, 0)
        self.assertEqual(res["n_avaliadas"], 19)
        self.assertEqual(len(res["problemas_de_preenchimento"]), 1)

    def test_id_desconhecido_e_ausente(self):
        out, linhas, _ = self._folha_preenchida()
        linhas[0]["id"] = "Q0000"
        rc, _, _, err = self._apura(self._escreve(linhas), out / "chave_oculta.json")
        self.assertEqual(rc, 2)
        self.assertIn("não está na chave", err)
        self.assertIn("não aparecem na folha", err)

    def test_valida_geral_s_com_dimensao_n_gera_aviso(self):
        out, linhas, _ = self._folha_preenchida()
        linhas[0]["valida_geral"], linhas[0]["resposta_unica"] = "s", "n"
        rc, res, _, err = self._apura(self._escreve(linhas), out / "chave_oculta.json")
        self.assertEqual(rc, 0)
        self.assertTrue(res["avisos"])
        self.assertIn("valida_geral=s", err)

    def test_falta_coluna_da_rubrica_aborta(self):
        out, linhas, _ = self._folha_preenchida()
        p = self.d / "sem_coluna.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            campos = [c for c in ah.COLUNAS_FOLHA if c != "valida_geral"]
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
        with self.assertRaises(SystemExit):
            self._apura(p, out / "chave_oculta.json")


if __name__ == "__main__":
    unittest.main()
