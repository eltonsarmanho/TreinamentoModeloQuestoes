"""Tarefa F, itens 1-2: pool do 4B, ordem estratificada, folha cega e apuração parcial.

A parte lógica usa dados sintéticos; o bloco `TestSaidasGeradas` confere os arquivos reais de
outputs/melhoria_v8/F/ quando existem (rode src/f_rotulados.py e src/f_pool_4b.py antes).
"""
import csv
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import amostra_humana as ah  # noqa: E402
import f_pool_4b as fp  # noqa: E402
import f_apurar_lotes as fal  # noqa: E402
from unidades_tematicas import UNIDADES  # noqa: E402

F = RAIZ / "outputs" / "melhoria_v8" / "F"


def itens_sinteticos(por_unidade=(12, 12, 12, 12, 12), seed=1):
    its = []
    for u, n in zip(UNIDADES, por_unidade):
        for k in range(n):
            its.append({"origem_pasta": f"p{k % 3}", "id_original": f"{u[:3]}{k}", "unidade": u,
                        "sig": f"{u}-{k}", "ano": "5º", "habilidade": "H01", "dificuldade": "Fácil",
                        "status_chave": "ok", "seed_geracao": k})
    random.Random(seed).shuffle(its)
    return its


class TestOrdem(unittest.TestCase):
    def test_todo_prefixo_multiplo_de_5_eh_balanceado(self):
        ordem = fp.ordem_estratificada(itens_sinteticos(), seed=9)
        self.assertEqual(len(ordem), 60)
        for k in range(5, 61, 5):
            c = fp.prefixo_balanceado(ordem, k)
            self.assertEqual(set(c.values()), {k // 5}, (k, c))

    def test_determinista_e_depende_da_seed(self):
        a = [i["id_original"] for i in fp.ordem_estratificada(itens_sinteticos(), 3)]
        b = [i["id_original"] for i in fp.ordem_estratificada(itens_sinteticos(seed=77), 3)]
        c = [i["id_original"] for i in fp.ordem_estratificada(itens_sinteticos(), 4)]
        self.assertEqual(a, b)  # independe da ordem de entrada
        self.assertNotEqual(a, c)

    def test_unidade_menor_esgota_e_o_resto_segue(self):
        ordem = fp.ordem_estratificada(itens_sinteticos((10, 4, 10, 10, 10)), seed=2)
        self.assertEqual(len(ordem), 44)
        # até a linha 5*4=20 o balanço é exato; depois a unidade pequena não aparece mais
        self.assertEqual(set(fp.prefixo_balanceado(ordem, 20).values()), {4})
        self.assertEqual(fp.prefixo_balanceado(ordem, 44)[UNIDADES[1]], 4)

    def test_posicao_da_unidade_nao_eh_ciclica(self):
        ordem = fp.ordem_estratificada(itens_sinteticos((40,) * 5), seed=5)
        padroes = {tuple(i["unidade"] for i in ordem[r * 5:(r + 1) * 5]) for r in range(40)}
        self.assertGreater(len(padroes), 5)  # ordem das unidades sorteada por rodada


class TestDedup(unittest.TestCase):
    def test_avaliados_e_duplicatas_no_pool(self):
        c = [{"sig": "a", "origem_pasta": "x", "id_original": "1"},
             {"sig": "b", "origem_pasta": "x", "id_original": "2"},
             {"sig": "a", "origem_pasta": "y", "id_original": "9"},
             {"sig": "c", "origem_pasta": "y", "id_original": "3"}]
        mant, desc = fp.separar_duplicatas(c, {"b": "amostra_3bracos:Q1"})
        self.assertEqual([m["id_original"] for m in mant], ["1", "3"])
        motivos = {d["id_original"]: (d["motivo_descarte"], d["duplica"]) for d in desc}
        self.assertEqual(motivos["2"], ("ja_avaliado", "amostra_3bracos:Q1"))
        self.assertEqual(motivos["9"], ("duplicata_no_pool", "x:1"))

    def test_linha_avaliada(self):
        self.assertFalse(fp.linha_avaliada({"correcao_matematica": " ", "comentario": "x"}))
        self.assertTrue(fp.linha_avaliada({"valida_geral": "s"}))


class TestLotesEIds(unittest.TestCase):
    def test_lotes_de_40(self):
        ls = fp.dividir_lotes(list(range(95)))
        self.assertEqual([len(x) for x in ls], [40, 40, 15])

    def test_ids_unicos_opacos_e_deterministas(self):
        a = fp.atribuir_ids(300, 1)
        self.assertEqual(len(set(a)), 300)
        self.assertTrue(all(i.startswith("Q") and len(i) == 5 for i in a))
        self.assertEqual(a, fp.atribuir_ids(300, 1))


def _montar_folha_e_chave(tmp, n_por_unidade=12, preenche=25, seed=1):
    its = fp.ordem_estratificada(itens_sinteticos((n_por_unidade,) * 5), seed)
    ids = fp.atribuir_ids(len(its), seed)
    linhas = []
    for it, i in zip(its, ids):
        base = {k: "" for k in fp.COLUNAS_FOLHA}
        base.update({"id": i, "ano": "5º", "habilidade": "H01", "enunciado": "e", "A": "1", "B": "2",
                     "C": "3", "D": "4", "E": "5", "resolucao": "r", "gabarito_modelo": "A"})
        linhas.append(base)
    chave = {"itens": {i: {"origem": "ft4b", "unidade": it["unidade"]} for it, i in zip(its, ids)}}
    rng = random.Random(0)
    for l in linhas[:preenche]:
        for k in fp.ROTULOS:
            l[k] = rng.choice("sn")
    caminho = Path(tmp) / "folha_preenchida.csv"
    fp.escrever_csv(caminho, linhas)
    (Path(tmp) / "chave.json").write_text(json.dumps(chave), encoding="utf-8")
    return caminho, Path(tmp) / "chave.json", linhas, chave


class TestApuracaoParcial(unittest.TestCase):
    def test_apurar_original_recusa_sem_flag_e_aceita_com_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            folha, chave, linhas, _ = _montar_folha_e_chave(tmp)
            err, out = StringIO(), StringIO()
            with redirect_stderr(err), redirect_stdout(out):
                rc_sem = ah.main(["apurar", "--folha", str(folha), "--chave", str(chave)])
                rc_com = ah.main(["apurar", "--folha", str(folha), "--chave", str(chave),
                                  "--ignorar-incompletas", "--saida", str(Path(tmp) / "ap.json")])
            self.assertEqual(rc_sem, 2)
            self.assertEqual(rc_com, 0)
            ap = json.loads((Path(tmp) / "ap.json").read_text(encoding="utf-8"))
            self.assertEqual(ap["n_avaliadas"], 25)

    def test_f_apurar_lotes_bate_com_o_apurar_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            folha, chave, linhas, ch = _montar_folha_e_chave(tmp)
            # divide em lotes de 40 como a entrega real; só o 1º tem linhas preenchidas
            lotes = fp.dividir_lotes(linhas, 40)
            caminhos = []
            for k, lote in enumerate(lotes, 1):
                p = Path(tmp) / f"folha_lote_{k:02d}_preenchida.csv"
                fp.escrever_csv(p, lote)
                caminhos.append(str(p))
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = fal.main(["--lotes", *caminhos, "--chave", str(chave), "--saida", str(Path(tmp) / "a.json")])
                ah.main(["apurar", "--folha", str(folha), "--chave", str(chave),
                         "--ignorar-incompletas", "--saida", str(Path(tmp) / "b.json")])
            self.assertEqual(rc, 0)
            a = json.loads((Path(tmp) / "a.json").read_text(encoding="utf-8"))
            b = json.loads((Path(tmp) / "b.json").read_text(encoding="utf-8"))
            self.assertEqual(a["geral"], b["geral"])
            self.assertEqual(a["por_unidade"], b["por_unidade"])
            self.assertEqual(a["n_avaliadas"], 25)
            self.assertEqual(a["n_em_branco_ignoradas"], 35)
            # 25 primeiras linhas = 5 rodadas completas -> 5 por unidade
            self.assertEqual(set(a["itens_por_unidade"].values()), {5})

    def test_linha_parcial_e_valor_invalido_sao_descartados_e_listados(self):
        with tempfile.TemporaryDirectory() as tmp:
            folha, chave, linhas, ch = _montar_folha_e_chave(tmp, preenche=10)
            linhas[0]["resposta_unica"] = ""          # parcial
            linhas[1]["dados_suficientes"] = "talvez"  # inválido
            fp.escrever_csv(folha, linhas)
            validas, branco, probs = fal.ler_lotes([str(folha)], ch)
            self.assertEqual(len(validas), 8)
            self.assertEqual(len(probs), 2)
            self.assertEqual(len(branco), 50)


@unittest.skipUnless((F / "folha_professor_4b_pendentes.csv").exists(), "rode src/f_pool_4b.py")
class TestSaidasGeradas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linhas = list(csv.DictReader(open(F / "folha_professor_4b_pendentes.csv", encoding="utf-8-sig")))
        cls.chave = json.loads((F / "chave" / "chave_oculta.json").read_text(encoding="utf-8"))
        cls.rel = json.loads((F / "pool_4b_relatorio.json").read_text(encoding="utf-8"))

    def test_mesmas_colunas_das_folhas_existentes(self):
        ref = next(csv.reader(open(RAIZ / "outputs/amostra_3bracos/folha_professor.csv", encoding="utf-8-sig")))
        self.assertEqual(list(self.linhas[0].keys()), ref)

    def test_ids_unicos_e_na_chave_e_rubrica_vazia(self):
        ids = [l["id"] for l in self.linhas]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.chave["itens"]))
        for l in self.linhas:
            self.assertTrue(all(not l[k].strip() for k in (*fp.ROTULOS, "comentario")))

    def test_chave_fora_da_pasta_do_professor_e_sem_dica_de_origem_nas_folhas(self):
        self.assertFalse((F / "chave_oculta.json").exists())
        for p in [F / "folha_professor_4b_pendentes.csv", *sorted(F.glob("folha_lote_*.csv"))]:
            txt = p.read_text(encoding="utf-8-sig").lower()
            for dica in ("ft4b", "prova_", "lora", "qwen", "nao_verificavel", "seed"):
                self.assertNotIn(dica, txt, (p.name, dica))

    def test_lotes_concatenados_sao_a_folha_inteira(self):
        lotes = []
        for p in sorted(F.glob("folha_lote_[0-9][0-9].csv")):
            lotes += list(csv.DictReader(open(p, encoding="utf-8-sig")))
        self.assertEqual([l["id"] for l in lotes], [l["id"] for l in self.linhas])
        self.assertTrue(all(len(list(csv.DictReader(open(p, encoding="utf-8-sig")))) == 40
                            for p in sorted(F.glob("folha_lote_[0-9][0-9].csv"))[:-1]))

    def test_todo_prefixo_de_40_estratificado_enquanto_ha_todas_as_unidades(self):
        ate = self.rel["balanceado_ate"]
        unidade = {i: v["unidade"] for i, v in self.chave["itens"].items()}
        for k in range(5, ate + 1, 5):
            c = Counter(unidade[l["id"]] for l in self.linhas[:k])
            self.assertEqual(set(c.values()), {k // 5}, (k, c))
        self.assertGreaterEqual(ate, 160)

    def test_nenhum_item_pendente_ja_foi_avaliado(self):
        import f_rotulados  # noqa: F401
        sigs = set()
        for l in (F / "rotulados.jsonl").read_text(encoding="utf-8").splitlines():
            sigs.add(json.loads(l)["sig"])
        from f_comum import assinatura, questao_de_linha
        for l in self.linhas:
            self.assertNotIn(assinatura(questao_de_linha(l)), sigs)


if __name__ == "__main__":
    unittest.main()
