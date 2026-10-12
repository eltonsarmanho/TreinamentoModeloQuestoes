"""Ciclo v6, invariante 8: saída truncada (finish_reason="length") nunca é aprovação.

Antes, um juiz Maritaca cujo JSON final ainda fechava valia mesmo cortado pelo
max_tokens (só ficava fora do cache). Agora validador (fase 1 e 2) e revisor
reprovam com erro "resposta_truncada", como o árbitro Gemini já fazia.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agentes_questoes as aq  # noqa: E402
from test_agentes_questoes import FASE2_OK, agentes_com, fase1, questao_boa, revisor_json, so_v  # noqa: E402


class FilaComFim:
    """Devolve (texto, finish_reason) em ordem."""

    def __init__(self, respostas):
        self.respostas = list(respostas)

    def chat_completion(self, messages, max_tokens, temperature):
        texto, fim = self.respostas.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto), finish_reason=fim)])


class TestJuizTruncado(unittest.TestCase):
    def test_fase1_truncada_com_json_valido_reprova(self):
        q = questao_boa(gab="B")
        v = agentes_com(FilaComFim([(fase1(q, so_v("B")), "length")])).validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(v["erro"], "resposta_truncada")

    def test_fase2_truncada_com_json_valido_reprova(self):
        q = questao_boa(gab="B")
        v = agentes_com(FilaComFim([(fase1(q, so_v("B")), "stop"), (FASE2_OK, "length")])).validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(v["erro"], "resposta_truncada")

    def test_revisor_truncado_com_json_valido_reprova(self):
        q = questao_boa(gab="B")
        r = agentes_com(FilaComFim([(revisor_json(so_v("B"), questao=q), "length")])).revisar(q, "5º", "H01", "d")
        self.assertFalse(r["veredito"])
        self.assertEqual(r["erro"], "resposta_truncada")

    def test_mesmas_respostas_completas_aprovam(self):
        """Controle: a reprovação acima vem do finish_reason, não do conteúdo."""
        q = questao_boa(gab="B")
        v = agentes_com(FilaComFim([(fase1(q, so_v("B")), "stop"), (FASE2_OK, "stop")])).validar(q)
        self.assertTrue(v["veredito"])
        r = agentes_com(FilaComFim([(revisor_json(so_v("B"), questao=q), "stop")])).revisar(q, "5º", "H01", "d")
        self.assertTrue(r["veredito"])

    def test_resposta_sem_finish_reason_continua_valendo(self):
        """Cache e clientes antigos não trazem finish_reason: não é truncagem."""
        self.assertFalse(aq.resposta_truncada({}))
        self.assertFalse(aq.resposta_truncada({"cache": True}))


if __name__ == "__main__":
    unittest.main()
