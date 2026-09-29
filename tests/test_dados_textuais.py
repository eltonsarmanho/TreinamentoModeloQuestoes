"""Subtemas de dados (tabela/gráfico): o app é texto puro, então os dados
devem vir escritos no enunciado. Casos reais do GGUF (outputs/diversidade_atual*.json)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import diversidade as dv  # noqa: E402
import distill_teacher as dt  # noqa: E402
import gerar_lote as gl  # noqa: E402

COM_DADOS = ("Na tabela estão os votos da turma:\nAzul: 12\nVermelho: 6\nVerde: 9\n"
             "Qual cor teve mais votos?")
SEM_DADOS = "No gráfico, qual cor representa o número de famílias com 2 filhos?"


def _q(enunciado):
    return {"enunciado": enunciado, "alternativas": {k: f"op{k}" for k in "ABCDE"},
            "resolucao_passo_a_passo": "Comparando os valores.", "resposta_correta": "A",
            "difficulty": "EASY"}


class TestFamiliaDados(unittest.TestCase):
    def test_prefixos_genericos(self):
        for s in ("tabela_simples", "tabela_dupla", "grafico_barras", "grafico_linhas", "histograma"):
            self.assertTrue(dv.exige_dados_textuais(s), s)
        for s in ("triangulo", "fracao", None, ""):
            self.assertFalse(dv.exige_dados_textuais(s), s)

    def test_sufixo_so_para_dados(self):
        base = {"subtema_rotulo": "x", "tipo_raciocinio_rotulo": "y", "contexto_rotulo": "z"}
        self.assertIn(dv.INSTRUCAO_DADOS_TEXTUAIS, dv.sufixo_prompt({**base, "subtema": "tabela_simples"}))
        self.assertNotIn("Dados:", dv.sufixo_prompt({**base, "subtema": "triangulo"}))

    def test_todas_habilidades_de_dados_recebem_instrucao(self):
        for ano, hab in (("5º", "H21"), ("5º", "H22"), ("2º", "H20"), ("9º", "H24")):
            for slot in dv.planejar_lote(ano, hab, 5, "Fácil", seed=1):
                if dv.exige_dados_textuais(slot["subtema"]):
                    self.assertIn("Dados:", dv.sufixo_prompt(slot))


class TestDadosAusentes(unittest.TestCase):
    def test_detecta(self):
        self.assertTrue(dv.dados_ausentes(_q(SEM_DADOS), "grafico_pictorico"))
        self.assertFalse(dv.dados_ausentes(_q(COM_DADOS), "tabela_simples"))

    def test_cita_grafico_sem_dados_em_qualquer_slot(self):
        self.assertTrue(dv.dados_ausentes(_q(SEM_DADOS), "triangulo"))

    def test_nao_pune_questao_sem_citar_artefato(self):
        self.assertFalse(dv.dados_ausentes(_q("Quanto é 3 + 4?"), "adicao"))

    def test_violacao_e_restricao(self):
        slot = dv.planejar_lote("5º", "H22", 1, "Fácil", seed=0)[0]
        viol = dv.violacoes_diversidade([], _q(SEM_DADOS), slot, "5º", "H22", 1)
        self.assertIn("dados_ausentes", [v["tipo"] for v in viol])
        self.assertIn("rótulo: valor", dv.montar_restricao(viol, "H22", slot))

    def test_ranking_prefere_resolvivel(self):
        a = {"obj": {}, "status": "ok", "violacoes": [{"tipo": "dados_ausentes"}]}
        b = {"obj": {}, "status": "ok", "violacoes": [{"tipo": "contexto_repetido"},
                                                       {"tipo": "near_duplicata"}]}
        self.assertGreater(gl._chave(b), gl._chave(a))


class TestFiltroDestilacao(unittest.TestCase):
    def _slot(self, sub):
        return {"subtema": sub, "subtema_rotulo": sub, "tipo_raciocinio_rotulo": "y",
                "contexto_rotulo": "z"}

    def test_grafico_com_dados_passa(self):
        q = _q("O gráfico de barras mostra os gols:\nAna: 4\nBia: 7\nCaio: 2\nQuem fez mais gols?")
        motivo = dt.filtrar({"questoes": [q]}, "", set(), slot=self._slot("grafico_barras"))
        self.assertIsNone(motivo)

    def test_grafico_sem_dados_rejeitado(self):
        motivo = dt.filtrar({"questoes": [_q(SEM_DADOS)]}, "", set(), slot=self._slot("grafico_barras"))
        self.assertEqual(motivo, "dados_ausentes")

    def test_fora_de_dados_mantem_filtro_conservador(self):
        q = _q("Veja o gráfico: Ana 4, Bia 7, Caio 2, Davi 5. Quem fez mais?")
        motivo = dt.filtrar({"questoes": [q]}, "", set(), slot=self._slot("triangulo"))
        self.assertEqual(motivo, "menciona_figura")


if __name__ == "__main__":
    unittest.main()


class TestFormatosReais(unittest.TestCase):
    def test_valor_rotulo_e_frase_com_tres_valores(self):
        for en in ("Gráfico mostra: 12 alunos gostam de matemática; 10 de ciências; 8 de artes.",
                   "Um vendedor vendeu 15 brinquedos no primeiro mês, 25 no segundo e 5 no terceiro."):
            self.assertFalse(dv.dados_ausentes(_q(en), "grafico_colunas"), en)

    def test_dois_valores_e_insuficiente_para_grafico(self):
        self.assertTrue(dv.dados_ausentes(
            _q("Uma loja vendeu 12 carrinhos e 10 bonecos. Qual gráfico mostra isso?"), "grafico_barras"))
