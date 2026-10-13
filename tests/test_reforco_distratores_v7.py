"""Reforço sintético v7 (src/reforco_distratores_v7.py): schema, consistência, sem contaminação com val*."""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import reforco_distratores_v7 as rd  # noqa: E402
from schema_utils import _UNICODE_MATH, check_consistency, check_structure, extract_questoes  # noqa: E402


class TestGenEscritaPorExtenso(unittest.TestCase):
    def test_cinco_alternativas_distintas_em_varias_seeds(self):
        for seed in range(50):
            rng = random.Random(seed)
            q = rd.gen_escrita_por_extenso(rng, (10, 999999), "Moderado")
            self.assertEqual(len(set(q["alternativas"].values())), 5, msg=q["alternativas"])

    def test_gabarito_bate_com_a_resolucao(self):
        rng = random.Random(1)
        q = rd.gen_escrita_por_extenso(rng, (1000, 99999), "Difícil")
        letra = q["resposta_correta"]
        self.assertIn(q["alternativas"][letra], q["resolucao_passo_a_passo"])

    def test_sem_operador_unicode(self):
        rng = random.Random(2)
        for _ in range(20):
            q = rd.gen_escrita_por_extenso(rng, (10, 999999), "Fácil")
            blob = json.dumps(q, ensure_ascii=False)
            self.assertFalse(any(ch in _UNICODE_MATH and not ch.isspace() for ch in blob))


class TestGeracaoCompleta(unittest.TestCase):
    def test_gerar_todos_passa_validacao(self):
        exemplos = rd.gerar_todos(num_per_skill=4, seed=7)
        problemas, verificaveis, ok = rd.validar(exemplos)
        self.assertEqual(problemas, 0)
        self.assertEqual(ok, verificaveis)
        self.assertGreater(verificaveis, 0)

    def test_cobre_as_tres_habilidades_alvo(self):
        exemplos = rd.gerar_todos(num_per_skill=2, seed=7)
        pares = {(e["meta"]["ano"], e["meta"]["habilidade"]) for e in exemplos}
        self.assertEqual(pares, {("5º", "H03"), ("5º", "H01"), ("9º", "H02")})

    def test_prompt_identico_ao_contrato_do_app(self):
        from extract_data import SYSTEM_PROMPT
        exemplos = rd.gerar_todos(num_per_skill=1, seed=7)
        for e in exemplos:
            self.assertEqual(e["messages"][0]["content"], SYSTEM_PROMPT)

    def test_meta_marca_origem_e_sintetico(self):
        exemplos = rd.gerar_todos(num_per_skill=1, seed=7)
        for e in exemplos:
            self.assertTrue(e["meta"]["sintetico"])
            self.assertEqual(e["meta"]["origem_v7"], "reforco_distratores")

    def test_schema_structure_e_consistencia_individual(self):
        exemplos = rd.gerar_todos(num_per_skill=2, seed=3)
        for e in exemplos:
            obj = json.loads(e["messages"][2]["content"])
            flags = check_structure(obj)
            self.assertTrue(flags["schema_completo"] and flags["resposta_valida"] and flags["alternativas_distintas"])
            for q in extract_questoes(obj):
                ok, _sug = check_consistency(q)
                self.assertIn(ok, (True, None))  # nunca inconsistente (False)


class TestProtecoesDeArquivo(unittest.TestCase):
    def test_nao_escreve_em_arquivo_protegido(self):
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "src" / "reforco_distratores_v7.py"),
                            "--saida", str(ROOT / "data" / "train_curado_v7.jsonl"), "--num-per-skill", "1"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ABORTADO", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
