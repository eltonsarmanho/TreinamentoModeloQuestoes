"""Texto da questão entregue no relatório de `test_model.batch` (P0-4, A).

Sem llama-cli: `generate` é mockado (padrão de test_integracao_geometria).
Cobre: campos novos em detalhes[], retrocompatibilidade com os consumidores
(promover_checkpoint lê relatório com e sem `obj`) e o custo em tamanho.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import promover_checkpoint as pc  # noqa: E402
import test_model as tm  # noqa: E402

VAL = RAIZ / "data" / "val_frozen_v1.jsonl"


def _roda_batch(exemplos, d, **kw):
    val = Path(d) / "val.jsonl"
    val.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in exemplos) + "\n",
                   encoding="utf-8")
    gguf = Path(d) / "m.gguf"
    gguf.write_bytes(b"x")
    out = Path(d) / "rel.json"
    respostas = [e["messages"][2]["content"] for e in exemplos]
    estado = {"i": 0}

    def falso_generate(llama_cli, g, prompt, threads, mnt, seed=None, grammar=None):
        r = respostas[min(estado["i"], len(respostas) - 1)]
        estado["i"] += 1
        return r, None, 30.0, 0.01

    with mock.patch.object(tm, "generate", side_effect=falso_generate), \
            mock.patch("builtins.print"):
        tm.batch("cli", gguf, 4, None, val_path=val, report_path=out, **kw)
    return json.loads(out.read_text(encoding="utf-8")), out


class TestTextoNoRelatorio(unittest.TestCase):
    def test_detalhes_trazem_obj_habilidade_status_e_regeneracoes(self):
        ex = {"messages": [{"role": "system", "content": "s"},
                           {"role": "user", "content": "Gere 1 questão 5º H03"},
                           {"role": "assistant", "content": json.dumps({"questoes": [{
                               "enunciado": "Quanto é 2 + 3?",
                               "alternativas": {"A": "4", "B": "5", "C": "6", "D": "7", "E": "8"},
                               "resolucao_passo_a_passo": "2 + 3 = 5",
                               "resposta_correta": "B", "difficulty": "EASY"}]})}],
              "meta": {"codigo_item": "X1", "ano": "5º", "habilidade": "H03",
                       "dificuldade": "Fácil"}}
        with tempfile.TemporaryDirectory() as d:
            rel, _ = _roda_batch([ex], d)
        det = rel["detalhes"][0]
        self.assertEqual(det["obj"]["enunciado"], "Quanto é 2 + 3?")
        self.assertEqual(det["obj"]["resposta_correta"], "B")
        self.assertEqual((det["habilidade"], det["status"], det["regeneracoes"]),
                         ("H03", "ok", 0))
        # nada do que os gates leem mudou de lugar
        self.assertEqual(rel["pos_processamento"]["ok"], 1)
        self.assertIn("consistencia_resposta_correta", det)

    def test_json_invalido_grava_obj_none(self):
        ex = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "p"},
                           {"role": "assistant", "content": "isto nao e json"}],
              "meta": {"codigo_item": "X2", "ano": "9º", "habilidade": "H01",
                       "dificuldade": "Fácil"}}
        with tempfile.TemporaryDirectory() as d:
            rel, _ = _roda_batch([ex], d)
        self.assertIsNone(rel["detalhes"][0]["obj"])
        self.assertEqual(rel["detalhes"][0]["status"], "falha")

    @unittest.skipUnless(VAL.exists(), "val congelado ausente")
    def test_tamanho_com_texto_real_do_val_e_consumidores(self):
        exemplos = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines()]
        with tempfile.TemporaryDirectory() as d:
            rel, caminho = _roda_batch(exemplos, d)
            com = caminho.stat().st_size
            self.assertTrue(all(x["obj"] for x in rel["detalhes"]))
            # o mesmo relatório sem os campos novos = formato antigo
            antigo = json.loads(json.dumps(rel))
            for x in antigo["detalhes"]:
                x.pop("obj"), x.pop("habilidade")
            sem = len(json.dumps(antigo, ensure_ascii=False, indent=2).encode())
        print(f"\n[tamanho] relatório de {len(exemplos)} itens: sem texto {sem} B -> "
              f"com texto {com} B (+{com - sem} B, +{100 * (com - sem) / sem:.0f}%)")
        self.assertGreater(com, sem)
        self.assertLess(com / sem, 3.0)  # ordem de grandeza: KBs, não MBs
        # promover_checkpoint lê os dois formatos sem tocar nos campos novos
        # (os dois lados são o mesmo texto: só importa que não levante).
        g_novo = pc.avalia(rel, rel)
        g_antigo = pc.avalia(antigo, antigo)
        self.assertEqual([(g.id, g.passou) for g in g_novo],
                         [(g.id, g.passou) for g in g_antigo])


class TestRelatoriosAntigosReais(unittest.TestCase):
    def test_gate_multiseed_com_relatorios_sem_texto(self):
        R = RAIZ / "outputs" / "relatorios"
        nomes = ["eval_baseline_s0_gguf", "eval_baseline_s1_gguf", "eval_baseline_s2_gguf",
                 "eval_v3_gguf", "eval_v3_s1_gguf", "eval_v3_s2_gguf"]
        if not all((R / f"{n}.json").exists() for n in nomes):
            self.skipTest("relatórios históricos ausentes")
        c = lambda n: json.loads((R / f"{n}.json").read_text(encoding="utf-8"))
        gates, ganhos, ab, ac = pc.avalia_multiseed([c(n) for n in nomes[:3]],
                                                    [c(n) for n in nomes[3:]])
        veredito = json.loads((R / "VEREDITO_v3_multiseed.json").read_text(encoding="utf-8"))
        # o veredito gravado é anterior ao G6* (agregado): compara os gates em comum
        atuais = {g.id: g.passou for g in gates}
        for g in veredito["gates"]:
            if g["id"] in atuais:
                self.assertEqual(atuais[g["id"]], g["passou"], g["id"])
        self.assertTrue(all(g.passou for g in gates if g.bloqueante))


if __name__ == "__main__":
    unittest.main()
