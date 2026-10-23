"""Variante Gemma 3: leitor de cabeçalho GGUF, formato de chat do SFT e gate sem dificuldade."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "src" / "gemma")]
sys.path.insert(0, str(RAIZ / "tests"))
import gemma_formato as gf  # noqa: E402
import gguf_meta  # noqa: E402
import promover_checkpoint as pc  # noqa: E402
from test_gate_multiseed import _gate, _rel  # noqa: E402


def _s(texto):
    b = texto.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(kvs):
    """kvs: lista de (chave, tipo, bytes_do_valor)."""
    corpo = b"".join(_s(k) + struct.pack("<I", t) + v for k, t, v in kvs)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + corpo


class TestGgufMeta(unittest.TestCase):
    def _arquivo(self, conteudo):
        f = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        f.write(conteudo)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_le_arquitetura_depois_de_escalar_e_array(self):
        arr = struct.pack("<IQ", 8, 2) + _s("a") + _s("bb")
        caminho = self._arquivo(_gguf([
            ("general.alignment", 4, struct.pack("<I", 32)),
            ("tokenizer.ggml.tokens", 9, arr),
            ("general.architecture", 8, _s("gemma3")),
            ("general.name", 8, _s("Gemma 3 1B")),
        ]))
        self.assertEqual(gguf_meta.arquitetura(caminho), "gemma3")
        self.assertEqual(gguf_meta.metadados(caminho)["general.name"], "Gemma 3 1B")

    def test_arquivo_que_nao_e_gguf_vira_none(self):
        self.assertIsNone(gguf_meta.arquitetura(self._arquivo(b"nao sou gguf")))
        self.assertIsNone(gguf_meta.arquitetura("/caminho/que/nao/existe.gguf"))


PROMPT = "<start_of_turn>user\nU<end_of_turn>\n<start_of_turn>model\n"
RESPOSTA = '{"questoes": []}'
TEXTO = PROMPT + RESPOSTA + "<end_of_turn>\n"


class TestFormatoGemma(unittest.TestCase):
    def test_exemplo_correto_nao_tem_problema(self):
        self.assertEqual(gf.problemas_formato(TEXTO, PROMPT, RESPOSTA), [])

    def test_bos_duplicado_e_detectado(self):
        erros = gf.problemas_formato("<bos>" + TEXTO, "<bos>" + PROMPT, RESPOSTA)
        self.assertTrue(any("<bos>" in e for e in erros))

    def test_prompt_de_geracao_diferente_do_treino_e_detectado(self):
        erros = gf.problemas_formato(TEXTO, PROMPT.replace("U", "X"), RESPOSTA)
        self.assertTrue(any("divergem" in e for e in erros))

    def test_sem_fim_de_turno_e_detectado(self):
        erros = gf.problemas_formato(PROMPT + RESPOSTA, PROMPT, RESPOSTA)
        self.assertTrue(any("<end_of_turn>" in e for e in erros))

    def test_remover_bos_e_diferenca(self):
        self.assertEqual(gf.remover_bos("<bos>abc"), "abc")
        self.assertIsNone(gf.diferenca("abc", "abc"))
        self.assertEqual(gf.diferenca("abc", "abd")["posicao"], 2)


def _com_familia(rel, familia):
    rel["familia_modelo"] = familia
    return rel


class TestGateGemma(unittest.TestCase):
    def test_familias_diferentes_abortam(self):
        bases = [_com_familia(_rel(r, "qwen"), "qwen3") for r in range(3)]
        cands = [_com_familia(_rel(r, "gemma"), "gemma3") for r in range(3)]
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed(bases, cands)

    def test_mesma_familia_passa(self):
        bases = [_com_familia(_rel(r, "zs"), "gemma3") for r in range(3)]
        cands = [_com_familia(_rel(r, "ft"), "gemma3") for r in range(3)]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertTrue(_gate(gates, "G2*").passou)

    def _dificuldade_pior(self):
        bases = [_rel(r, "zs") for r in range(3)]
        cands = [_rel(r, "ft", inconsistentes=()) for r in range(3)]
        for c in cands:
            c["estrutura"]["difficulty_aderente_pct"] = 50.0
        return bases, cands

    def test_padrao_mantem_g5_bloqueante(self):
        gates, _, _, _ = pc.avalia_multiseed(*self._dificuldade_pior())
        g5 = _gate(gates, "G5")
        self.assertFalse(g5.passou)
        self.assertTrue(g5.bloqueante)

    def test_sem_dificuldade_deixa_g5_informativo(self):
        gates, _, _, _ = pc.avalia_multiseed(*self._dificuldade_pior(), sem_dificuldade=True)
        g5 = _gate(gates, "G5")
        self.assertFalse(g5.passou)
        self.assertFalse(g5.bloqueante)
        self.assertEqual([g.id for g in gates if g.bloqueante and not g.passou], [])

    def test_sem_dificuldade_tira_aderencia_dos_ganhos(self):
        bases = [_rel(r, "zs") for r in range(3)]
        cands = [_rel(r, "ft") for r in range(3)]
        for b in bases:
            for d in b["detalhes"]:
                d["difficulty_aderente"] = False
        _, ganhos, _, _ = pc.avalia_multiseed(bases, cands)
        _, ganhos_sem, _, _ = pc.avalia_multiseed(bases, cands, sem_dificuldade=True)
        self.assertTrue(any("dificuldade" in g for g in ganhos))
        self.assertFalse(any("dificuldade" in g for g in ganhos_sem))


if __name__ == "__main__":
    unittest.main()
