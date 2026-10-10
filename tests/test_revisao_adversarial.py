"""Regressões da revisão adversarial das Fases 0 e 1 (2026-09-30).

Cada teste aqui reproduz um achado CONFIRMADO da revisão e falha na versão
anterior do código. As questões são copiadas LITERALMENTE dos corpora reais do
projeto (data/*.jsonl e outputs/diversidade_*.json); a origem está no docstring
de cada uma.
"""
import json
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import re                                                # noqa: E402

import gerar_lote as gl                                  # noqa: E402
import promover_checkpoint as pc                         # noqa: E402
import schema_utils as su                                # noqa: E402


# ---------------------------------------------------------------------------
# ACHADO 1 (alto) — resposta_fora_das_alternativas acusa questões CORRETAS, e a
# Fase 1 usa esse veredito como rejeição dura (score 2, status "falha").
# ---------------------------------------------------------------------------

# data/train_curado.jsonl:430 — média aritmética com arredondamento. Gabarito
# B="7.8" está CERTO; o verificador lia só "31 ÷ 4 = 7,75" e parava ali.
Q_ARREDONDA = {
    "enunciado": "Qual é a média aritmética dessas notas? Notas: 7,5 - 6,0 - 8,5 - 9,0",
    "alternativas": {"A": "7.5", "B": "7.8", "C": "8", "D": "7.9",
                     "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo":
        "Média = (7,5 + 6,0 + 8,5 + 9,0) ÷ 4 = 31 ÷ 4 = 7,75 → arredondado = 7,8",
    "resposta_correta": "B",
    "difficulty": "HARD",
}

# data/train_curado.jsonl:492 — cadeia de 3+ operandos, que _EXPR_PATTERN não
# casa. Gabarito A="3" está CERTO.
Q_CADEIA = {
    "enunciado": "Considere a expressão: x² - 4x + 6. Sabendo que x = 3, qual é o valor?",
    "alternativas": {"A": "3", "B": "4", "C": "5", "D": "6",
                     "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo": "3² - 4x3 + 6 = 9 - 12 + 6 = 3",
    "resposta_correta": "A",
    "difficulty": "HARD",
}

# data/val.jsonl:12 — moeda entre o "=" e o número. Gabarito C está CERTO.
Q_MOEDA = {
    "enunciado": "Um produto de R$ 900,00 tem 15% de desconto e depois 5% de "
                 "acréscimo, parcelado em 4 vezes. Qual o valor da parcela?",
    "alternativas": {"A": "R$ 225,00", "B": "R$ 191,25", "C": "R$ 212,25",
                     "D": "R$ 200,00", "E": "R$ 236,25"},
    "resolucao_passo_a_passo":
        "Desconto de 15% sobre 900 = 135. 900 - 135 = 765. Acréscimo de 5% "
        "sobre 765 = 765 x 1,05 = 803,25. Parcelas = 803,25 ÷ 4 = R$ 212,25",
    "resposta_correta": "C",
    "difficulty": "dificil",
}

# outputs/diversidade_exp_C_s2.json::ajustado::P12-5º-H21-N5::q4 — ERRO REAL.
# A resolução conclui 70 e nenhuma alternativa vale 70; o gabarito C=110 é
# indefensável. Tem de CONTINUAR sendo acusado.
Q_ERRO_REAL = {
    "enunciado": "A barraca que mais vendeu arrecadou 150 reais e a que menos "
                 "vendeu, 80 reais. Qual é a diferença?",
    "alternativas": {"A": "60", "B": "100", "C": "110", "D": "120", "E": "130"},
    "resolucao_passo_a_passo":
        "O maior valor é 150 (Pastel) e o menor valor é 80 (Cachorro-Quente). "
        "Calculando a diferença: 150 - 80 = 70. Portanto, a diferença é 70 unidades.",
    "resposta_correta": "C",
    "difficulty": "medio",
}


class TestForaDasAlternativasNaoAcusaQuestaoCorreta(unittest.TestCase):
    """REGRA E (guarda de cauda). Antes da correção as três questões corretas
    abaixo devolviam (False, None, 'resposta_fora_das_alternativas'), o que na
    Fase 1 vale score 2 — abaixo de QUALQUER questão não verificável (6) — e
    status 'falha', que zera o gate G2."""

    def test_arredondamento_nao_e_acusado(self):
        ok, sug, motivo = su.check_consistency_detalhado(Q_ARREDONDA)
        self.assertIsNot(ok, False, f"falso positivo (motivo={motivo})")
        self.assertFalse(su.resposta_fora_das_alternativas(Q_ARREDONDA))

    def test_cadeia_de_tres_operandos_nao_e_acusada(self):
        ok, sug, motivo = su.check_consistency_detalhado(Q_CADEIA)
        self.assertIsNot(ok, False, f"falso positivo (motivo={motivo})")
        self.assertFalse(su.resposta_fora_das_alternativas(Q_CADEIA))

    def test_moeda_entre_igual_e_numero_nao_e_acusada(self):
        ok, sug, motivo = su.check_consistency_detalhado(Q_MOEDA)
        self.assertIsNot(ok, False, f"falso positivo (motivo={motivo})")
        self.assertFalse(su.resposta_fora_das_alternativas(Q_MOEDA))

    def test_a_guarda_nao_absolve_erro_real(self):
        """Discriminação: a guarda tem de ser uma absolvição estreita, não um
        desligamento do detector."""
        ok, sug, motivo = su.check_consistency_detalhado(Q_ERRO_REAL)
        self.assertIs(ok, False)
        self.assertEqual(motivo, su.MOTIVO_FORA_DAS_ALTERNATIVAS)
        self.assertTrue(su.resposta_fora_das_alternativas(Q_ERRO_REAL))

    def test_fix_gabarito_deixa_a_questao_correta_intacta_e_sem_rotulo_de_falha(self):
        q, status = su.fix_gabarito(dict(Q_ARREDONDA))
        self.assertNotEqual(status, "fora_das_alternativas")
        self.assertEqual(q["resposta_correta"], "B")

    def test_guarda_so_absolve_nunca_acusa(self):
        """_valor_na_cauda é consultada por ÚLTIMO, depois de todas as outras
        absolvições, e o único veredito que ela pode reescrever é False.
        Sensor: forçá-la a True não muda nenhum veredito que não fosse False —
        se ela pudesse acusar, este teste pegaria."""
        consistentes = [Q_MOEDA, Q_CADEIA, Q_ARREDONDA, {
            "enunciado": "Quanto é 12 + 8?",
            "alternativas": {"A": "20", "B": "18", "C": "22", "D": "16",
                             "E": "Nenhuma das alternativas anteriores"},
            "resolucao_passo_a_passo": "12 + 8 = 20",
            "resposta_correta": "A", "difficulty": "facil"}]
        original = su._valor_na_cauda
        try:
            su._valor_na_cauda = lambda *a, **k: False   # guarda DESLIGADA
            antes = [su.check_consistency(q)[0] for q in consistentes]
            su._valor_na_cauda = lambda *a, **k: True    # guarda SEMPRE ativa
            depois = [su.check_consistency(q)[0] for q in consistentes]
        finally:
            su._valor_na_cauda = original
        for a, d, q in zip(antes, depois, consistentes):
            with self.subTest(enunciado=q["enunciado"][:40]):
                if a is not False:
                    self.assertIs(d, a, "a guarda alterou um veredito não-False")
                else:
                    self.assertIsNone(d, "False só pode virar None")


# ---------------------------------------------------------------------------
# ACHADO 2 (crítico) — o veto V8 deixava permutar questões cuja resolução nomeia
# a letra do gabarito, entregando ao aluno uma questão que se contradiz.
# ---------------------------------------------------------------------------

Q_CITA_CANONICA = {
    "enunciado": "Um circulo tem raio 5 cm. Qual o diametro?",
    "alternativas": {"A": "5 cm", "B": "10 cm", "C": "15 cm", "D": "20 cm",
                     "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo":
        "O diametro vale o dobro do raio. Portanto, a alternativa correta e B.",
    "resposta_correta": "B",
    "difficulty": "facil",
}

# outputs/diversidade_exp_C_s1.json::ajustado::P04-9º-H18-N5 — citação com
# inicial MAIÚSCULA, que escapava por falta de re.IGNORECASE nas palavras.
Q_CITA_MAIUSCULA = {
    "enunciado": "Um quadrado tem perimetro de 64 cm. Qual o lado?",
    "alternativas": {"A": "8 cm", "B": "32 cm", "C": "12 cm", "D": "16 cm",
                     "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo":
        "O lado vale o perimetro dividido por 4. Alternativa D (16 cm) e correta.",
    "resposta_correta": "D",
    "difficulty": "facil",
}


class TestVetoDeCitacaoDeLetra(unittest.TestCase):
    def test_forma_canonica_e_vetada(self):
        """'a alternativa correta é B' — a forma mais comum em português, que o
        padrão antigo não pegava porque exigia a letra COLADA ao substantivo."""
        self.assertIn("resolucao_cita_letra", gl.vetos_permutacao(Q_CITA_CANONICA))

    def test_inicial_maiuscula_e_vetada(self):
        """'Alternativa D ... é correta' — início de frase, que é onde a citação
        mais aparece, e escapava por falta de case-insensitivity nas palavras."""
        self.assertIn("resolucao_cita_letra", gl.vetos_permutacao(Q_CITA_MAIUSCULA))

    def test_questao_citante_nao_e_permutada(self):
        for q in (Q_CITA_CANONICA, Q_CITA_MAIUSCULA):
            with self.subTest(gab=q["resposta_correta"]):
                for seed in range(8):
                    p = gl.permutar_alternativas(q, seed)
                    self.assertEqual(p["resposta_correta"], q["resposta_correta"])
                    self.assertEqual(p["alternativas"], q["alternativas"])

    def test_case_insensitive_nao_vaza_para_a_letra(self):
        """A flag tem de ser LOCAL às palavras: se `[A-E]` virasse
        case-insensitive, `\\b[A-E]\\b` passaria a casar o artigo 'a', a
        conjunção 'e' e a preposição 'o', vetando praticamente todo o corpus."""
        limpos = [
            "Somando 3 + 4 = 7, o perimetro e 7 cm.",
            "A area do time A e maior que a do time B.",
            "O dobro de 6 e 12, entao o total e 12 reais.",
        ]
        for texto in limpos:
            with self.subTest(texto=texto[:30]):
                self.assertIsNone(gl._RESOL_CITA_LETRA.search(texto))

    def test_resolucao_sem_citacao_continua_permutavel(self):
        """A correção não pode vetar o corpus inteiro: sem citação, permuta."""
        q = dict(Q_CITA_CANONICA,
                 resolucao_passo_a_passo="O diametro vale o dobro do raio: 5 x 2 = 10 cm.")
        self.assertEqual(gl.vetos_permutacao(q), [])
        self.assertNotEqual(gl.permutar_alternativas(q, 1)["resposta_correta"], "B")


# ---------------------------------------------------------------------------
# ACHADO 3 (alto) — a âncora posicional só era reconhecida na forma FEMININA, e
# a masculina era embaralhada para o meio/início da lista.
# ---------------------------------------------------------------------------

class TestAncoraMasculina(unittest.TestCase):
    ANCORAS = [
        "Nenhum dos anteriores",
        "Nenhum dos três",
        "Nenhum dos quadriláteros anteriores",
        "Nenhuma das alternativas anteriores",   # controle: já funcionava
        "Todos os anteriores",
    ]

    def test_reconhecidas(self):
        for texto in self.ANCORAS:
            with self.subTest(texto=texto):
                self.assertTrue(gl._ANCORA_PATTERN.search(texto))

    def test_nao_reconhece_alternativa_comum(self):
        for texto in ("12 cm", "Maria", "O triângulo é retângulo.", "2/3"):
            with self.subTest(texto=texto):
                self.assertIsNone(gl._ANCORA_PATTERN.search(texto))

    def test_ancora_masculina_e_congelada_na_permutacao(self):
        """outputs/diversidade_candidato.json::ajustado::lote11::q3 — a âncora
        em E ia para A, entregando 'A) Nenhum dos anteriores' seguida das opções
        que o aluno deveria descartar."""
        q = {
            "enunciado": "Qual disciplina teve mais votos na pesquisa?",
            "alternativas": {"A": "Ciências", "B": "Matemática", "C": "Artes",
                             "D": "Educação Física", "E": "Nenhum dos anteriores"},
            "resolucao_passo_a_passo":
                "Matemática recebeu 18 votos, o maior total da tabela.",
            "resposta_correta": "B",
            "difficulty": "facil",
        }
        for seed in range(12):
            with self.subTest(seed=seed):
                for p in (gl.permutar_alternativas(q, seed),
                          gl.permutar_lote([q], seed)[0]):
                    self.assertEqual(p["alternativas"]["E"], "Nenhum dos anteriores")
                    self.assertEqual(p["alternativas"][p["resposta_correta"]],
                                     "Matemática")


# ---------------------------------------------------------------------------
# ACHADO 4 (crítico) — exigir p<ALFA para reprovar desarmava P3 e G11 no único
# regime em que eles operam (McNemar exato bicaudal: p mínimo 2/2^d).
# ---------------------------------------------------------------------------

class TestGatesReprovamDeterioracaoPequena(unittest.TestCase):
    def _rel(self, sha, inconsistentes_por_lote, n_por_lote=76, lotes=3):
        ls = []
        for i in range(lotes):
            questoes = []
            for j in range(n_por_lote):
                ruim = j < inconsistentes_por_lote
                questoes.append({
                    "enunciado": f"Quanto e {j} mais {j}?",
                    "alternativas": {"A": str(2 * j), "B": str(2 * j + 1),
                                     "C": str(2 * j + 2), "D": str(2 * j + 3),
                                     "E": "Nenhuma das alternativas anteriores"},
                    "resolucao_passo_a_passo": f"{j} + {j} = {2 * j}",
                    "resposta_correta": "B" if ruim else "A",
                    "difficulty": "facil",
                })
            ls.append({
                "id": f"P{i}", "ano": "5º", "habilidade": "H17",
                "seed_pareamento": i, "quantidade_pedida": n_por_lote,
                "quantidade_gerada": n_por_lote, "chamadas": 1, "json_validos": 1,
                "schema_ok": n_por_lote, "aderentes": n_por_lote,
                "aderencia_mensuravel": False, "depende_de_visual": 0,
                "difficulty_correta": n_por_lote, "questoes": questoes,
                "consistencia": {"ok": n_por_lote - inconsistentes_por_lote,
                                 "inconsistente": inconsistentes_por_lote,
                                 "nao_verificavel": 0},
            })
        total = n_por_lote * lotes
        ruins = inconsistentes_por_lote * lotes
        return {
            "artefato_sha256": sha, "prompt_ids": [f"P{i}" for i in range(lotes)],
            "modos": {"ajustado": {"lotes": ls, "agregado": {
                "consistencia_inconsistente_pct": round(100.0 * ruins / total, 2),
                "json_valido_pct": 100.0, "schema_pct": 100.0,
                "diversity_score": 0.9,
            }}},
        }

    def test_piora_pequena_acima_da_tolerancia_reprova_p3_e_g11(self):
        """MEDIDO na revisão: com a régua anterior, quebrar até 5 gabaritos em
        228 passava por P3 E por G11 (p=0,0625 >= 0,05), mesmo com a métrica em
        2,7x o baseline. Aqui: 0 -> 5 em 228 = 0,0% -> 2,19%, mais que o dobro
        da tolerância de 1pp."""
        base = self._rel("a", 0)
        cand = self._rel("b", 0)
        quebrados = 0
        for lote in cand["modos"]["ajustado"]["lotes"]:
            for q in lote["questoes"]:
                if quebrados < 5 and q["resposta_correta"] == "A":
                    q["resposta_correta"] = "C"
                    lote["consistencia"]["inconsistente"] += 1
                    lote["consistencia"]["ok"] -= 1
                    quebrados += 1
        self.assertEqual(quebrados, 5)
        cand["modos"]["ajustado"]["agregado"]["consistencia_inconsistente_pct"] = 2.19
        mc = pc.mcnemar_planejado(base, cand, "ajustado",
                                  "consistencia_inconsistente_pct")
        self.assertIsNotNone(mc)
        piorou, melhorou, p, n, fora = mc
        self.assertEqual((piorou, melhorou, fora), (5, 0, 0))
        self.assertGreaterEqual(p, pc.ALFA, "5 discordantes nunca atingem p<0,05")

        reprova, _txt, estourou = pc._piorou_de_verdade(
            base, cand, "ajustado", "consistencia_inconsistente_pct",
            0.0, 2.19, -1, pc.MARGENS_PLANEJADO["P3"][2])
        self.assertTrue(estourou)
        self.assertTrue(reprova, "p>=ALFA não pode absolver por falta de poder")
        self.assertFalse(pc.gate_diversidade(base, cand).passou)

    def test_limite_superior_alarga_quando_falta_poder(self):
        """A propriedade que sustenta a correção: com pouca amostra o limite
        superior CRESCE (nega a absolvição), enquanto o p-valor CRESCE também
        (concedia a absolvição). Direções opostas — só uma serve a um gate
        bloqueante."""
        self.assertGreater(pc.limite_superior_piora(3, 0, 228),
                           pc.limite_superior_piora(3, 0, 2280))
        self.assertGreater(pc.limite_superior_piora(5, 0, 228), 1.0)
        self.assertIsNone(pc.limite_superior_piora(0, 0, 0))

    def test_sem_discordancia_absolve(self):
        """A absolvição não sumiu: zero pares discordantes é prova de
        não-inferioridade, e aí o limite superior é 0."""
        self.assertEqual(pc.limite_superior_piora(0, 0, 228), 0.0)


# ---------------------------------------------------------------------------
# Propriedade sobre o CORPUS REAL. outputs/ é gitignored: pulado quando ausente.
# ---------------------------------------------------------------------------

def _corpus():
    qs = []
    for p in sorted((RAIZ / "outputs").glob("diversidade_*.json")):
        if "dryrun" in p.name:
            continue
        try:
            rel = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for modo in (rel.get("modos") or {}).values():
            for lote in modo.get("lotes", []):
                qs.append(lote.get("questoes", []))
    return qs


class TestCorpusReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lotes = _corpus()
        if sum(len(l) for l in cls.lotes) < 200:
            raise unittest.SkipTest("outputs/diversidade_*.json ausente")

    # Detector INDEPENDENTE do veto que está sendo testado: usar
    # gl._RESOL_CITA_LETRA aqui tornaria o teste circular (uma regressão no veto
    # regrediria o detector junto e o teste continuaria passando).
    CITA = re.compile(
        r"(?i:alternativas?|letras?|op[çc](?:[ãa]o|[õo]es)|itens?|respostas?"
        r"|afirma[çc](?:[ãa]o|[õo]es))[^.\n]{0,24}?\b([A-E])\b")

    def test_nenhuma_questao_entregue_aponta_a_letra_antiga(self):
        """Antes da correção: 10 questões saíam com a resolução nomeando uma
        letra que já não era a resposta."""
        falhas = []
        for lote in self.lotes:
            for antes, depois in zip(lote, gl.permutar_lote(lote, 12345)):
                if antes.get("resposta_correta") == depois.get("resposta_correta"):
                    continue
                res = antes.get("resolucao_passo_a_passo") or ""
                if any(m.group(1) == antes.get("resposta_correta")
                       for m in self.CITA.finditer(res)):
                    falhas.append(antes.get("enunciado", "")[:60])
        self.assertEqual(falhas, [], f"{len(falhas)} questões contraditórias")

    def test_nenhuma_ancora_deslocada(self):
        falhas = []
        for lote in self.lotes:
            for antes, depois in zip(lote, gl.permutar_lote(lote, 12345)):
                alts = antes.get("alternativas") or {}
                for L, txt in alts.items():
                    if (gl._ANCORA_PATTERN.search(str(txt))
                            and (depois.get("alternativas") or {}).get(L) != txt):
                        falhas.append((L, str(txt)[:40]))
        self.assertEqual(falhas, [], f"{len(falhas)} âncoras deslocadas")

    def test_permutacao_ainda_corrige_o_vies(self):
        """A correção aperta os vetos; tem de continuar valendo a pena."""
        from collections import Counter
        antes, depois = Counter(), Counter()
        for lote in self.lotes:
            for a, d in zip(lote, gl.permutar_lote(lote, 12345)):
                antes[a.get("resposta_correta")] += 1
                depois[d.get("resposta_correta")] += 1
        total = sum(antes.values())
        self.assertGreater(100 * max(antes.values()) / total, 35.0)
        self.assertLess(100 * max(depois.values()) / total, 30.0)

    def test_precisao_do_veredito_fora_das_alternativas(self):
        """Depois da Regra E restavam 10 instâncias acusadas nos relatórios (eram
        12), e as 2 que saíram são os casos de arredondamento discutível.

        2026-10-10 (verificador estendido, ver schema_utils._analisar_contas): o
        teto "<= 10" media a COBERTURA da regex binária, não a precisão. Com
        somas n-árias/unidades/encadeadas o mesmo veredito passou a pegar 79
        instâncias (36 questões únicas novas); as 36 foram lidas à mão e são erros
        reais do gerador (ex.: "5 + 8 + 8 + 5 = 26" com alternativas 10..30). O que
        este teste protege agora é (a) NADA se perdeu: as 10 antigas continuam
        acusadas, e (b) o veredito não explodiu (teto de sanidade)."""
        n = sum(1 for lote in self.lotes for q in lote
                if su.resposta_fora_das_alternativas(q))
        self.assertGreaterEqual(n, 10)
        self.assertLessEqual(n, 100)


if __name__ == "__main__":
    unittest.main()
