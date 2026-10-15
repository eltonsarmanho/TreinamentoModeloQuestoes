"""2026-10-14: val.jsonl e val_frozen_v1.jsonl têm os mesmos itens em ordem diferente e a seed
de geração deriva do índice da linha. Nenhum script de avaliação pode usar val.jsonl como padrão."""
import json
import re
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))


def _codigos(nome):
    return [json.loads(l)["meta"]["codigo_item"] for l in open(RAIZ / "data" / nome, encoding="utf-8")]


class TestValCongeladoComoPadrao(unittest.TestCase):
    def test_padrao_do_harness_e_do_evaluate_e_o_congelado(self):
        import test_model
        self.assertEqual(test_model.VAL_PATH.name, "val_frozen_v1.jsonl")
        src = (RAIZ / "src" / "evaluate.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'VAL_PATH\s*=\s*ROOT\s*/\s*"data"\s*/\s*"val_frozen_v1\.jsonl"')

    def test_nenhum_script_de_avaliacao_usa_val_jsonl_como_padrao(self):
        for arq in ("tests/test_model.py", "src/evaluate.py", "src/promover_checkpoint.py"):
            src = (RAIZ / arq).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r'VAL_PATH\s*=.*"val\.jsonl"', src), arq)

    def test_documenta_que_val_jsonl_nao_e_intercambiavel(self):
        """Mesmo conjunto, ordem diferente: se alguém regerar um dos dois, este teste avisa."""
        a, b = _codigos("val.jsonl"), _codigos("val_frozen_v1.jsonl")
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(a), len(b))


if __name__ == "__main__":
    unittest.main()
