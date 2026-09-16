"""Valida uma entrega de questões contra as regras do GUIA_ENTREGA_ALUNOS.md.

Este é o `validar_entrega.py` que o guia manda o aluno rodar antes de entregar
("só entregue quando sair 0 erros"). Ele não existia no repositório, então o
gate documentado era, na prática, inverificável: o professor recebia a entrega
sem nenhuma forma reproduzível de conferir o que o guia exige.

Fonte de verdade das regras: GUIA_ENTREGA_ALUNOS.md (Parte 3 + checklist final).
Fonte de verdade do contrato do modelo: schema_utils.py — a entrega é convertida
para o schema do app e submetida ao MESMO check_structure()/check_consistency()
que julga a saída do modelo em produção. Item que o verificador do app reprovaria
não entra no banco.

Duas classes de achado:
  ERRO   — bloqueia a incorporação. O item não entra no banco.
  AVISO  — não bloqueia, mas registra perda de qualidade (ex.: item sem conta
           reconhecível entra no treino sem poder ser conferido por máquina).

Uso:
    python src/validar_entrega.py TreinamentoDados/
    python src/validar_entrega.py TreinamentoDados/ --json relatorio.json
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from schema_utils import (
    DIFFICULTY_MAP,
    IMAGE_PATTERN,
    check_consistency,
    check_structure,
    normalize_math,
)
from extract_data import ALTERNATIVA_E_PADRAO

CAMPOS_OBRIGATORIOS = [
    "id_local", "codigo_bncc", "ano", "descricao_item", "disciplina", "tipo",
    "grau_resolucao", "leitura_enunciado", "enunciado_item", "texto_auxiliar",
    "alternativas", "gabarito", "justificativa_geral", "justificativas",
    "depende_de_imagem", "imagem", "descricao_imagem", "alternativas_imagem",
    "origem", "autor", "revisado_em",
]
GRAUS = {"Fácil", "Moderado", "Difícil"}
ANOS = {f"{i}º" for i in range(1, 10)}
LETRAS = "ABCD"
MAX_CHARS_RESPOSTA = 900       # limite de contexto do treino (guia, Parte 2)
MAX_FRACAO_GABARITO = 0.35     # "nenhuma letra em mais de 35% do seu lote"

# Operadores tipográficos que o Google Docs insere sozinho. O guia exige ASCII:
# o banco antigo misturava as duas formas e o modelo aprendeu a inconsistência.
UNICODE_OPS = set("−–—×⋅∙∕＝")

# Conta "a op b = r" — a MESMA expressão que schema_utils._EXPR_PATTERN usa para
# verificar a saída do modelo. Contar ocorrências aqui é o que implementa a regra
# 2 do guia ("a conta tem que ser UMA SÓ"): o verificador lê a primeira e para,
# então uma segunda conta no texto silenciosamente reprova o item mais tarde.
EXPR = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*([-+xX*÷/])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)"
)
NUM_INICIAL = re.compile(r"\d+(?:[.,]\d+)?")

# Perguntas que pedem a IDENTIFICAÇÃO de uma entidade (pessoa, dia, mês, lugar).
# Respondê-las com uma quantidade é defeito semântico: o gate mecânico do guia
# não pega, porque a alternativa continua sendo texto válido começando por
# número. Ver RELATORIO_QUALIDADE — foi o único defeito sistêmico do lote de
# 2026-09 e concentrava-se inteiro numa habilidade não-aritmética (EF03MA01).
PERGUNTA_IDENTIFICACAO = re.compile(
    r"\b(quem|qual (?:delas|deles|dela|dele)|em qual (?:dia|mes|semana|ano|"
    r"biblioteca|loja|turma|caixa|time|mês)|qual foi o vencedor)\b", re.I
)
# ...a menos que a pergunta peça explicitamente um número.
PERGUNTA_QUANTIDADE = re.compile(
    r"\b(quantos|quantas|qual (?:e|é) (?:o|a) (?:numero|número|valor|resultado|"
    r"total|preco|preço|quantia)|qual (?:o|a) (?:valor|numero|número|preco|"
    r"preço|resultado|total|maior quantidade|menor quantidade)|"
    r"qual foi a (?:maior|menor) quantidade)\b", re.I
)


def _deacc(texto):
    s = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _norm_chave(texto):
    return re.sub(r"[^a-z0-9]+", " ", _deacc(texto)).strip()


class Achados:
    """Coleta erros e avisos por item, preservando o motivo de cada um."""

    def __init__(self):
        self.erros = defaultdict(list)
        self.avisos = defaultdict(list)

    def erro(self, item, codigo, detalhe=""):
        self.erros[item].append((codigo, detalhe))

    def aviso(self, item, codigo, detalhe=""):
        self.avisos[item].append((codigo, detalhe))


def _questao_no_contrato(item):
    """Converte o item da entrega para o schema do app (5 alternativas, A-E).

    Mesma transformação de extract_data.build_example(): é assim que o item vai
    aparecer no treino, então é assim que ele precisa ser validado.
    """
    alternativas = {
        letra: normalize_math(str(item["alternativas"][letra]).strip())
        for letra in LETRAS
    }
    alternativas["E"] = ALTERNATIVA_E_PADRAO
    enunciado = f"{item['enunciado_item']} {item.get('texto_auxiliar') or ''}".strip()
    return {
        "enunciado": normalize_math(enunciado),
        "alternativas": alternativas,
        "resolucao_passo_a_passo": normalize_math(
            item["justificativas"][item["gabarito"]]
        ),
        "resposta_correta": item["gabarito"],
        "difficulty": DIFFICULTY_MAP.get(item["grau_resolucao"], "MEDIUM"),
    }


def _valida_campos_nan(item, nome, achados):
    def anda(valor, caminho=""):
        if isinstance(valor, dict):
            for chave, sub in valor.items():
                anda(sub, f"{caminho}.{chave}")
        elif isinstance(valor, list):
            for i, sub in enumerate(valor):
                anda(sub, f"{caminho}[{i}]")
        elif isinstance(valor, str) and valor.strip().lower() == "nan":
            achados.erro(nome, "valor_nan", caminho.lstrip("."))

    anda(item)


def valida_item(item, nome, habilidade, dir_figuras, achados):
    """Roda todas as regras sobre um item. Retorna o dict do item ou None."""
    faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in item]
    if faltando:
        achados.erro(nome, "campo_ausente", ",".join(faltando))
        return None

    _valida_campos_nan(item, nome, achados)

    # --- metadados e enums -------------------------------------------------
    if item["ano"] not in ANOS:
        achados.erro(nome, "ano_invalido", repr(item["ano"]))
    if item["grau_resolucao"] not in GRAUS:
        achados.erro(nome, "grau_resolucao_invalido", repr(item["grau_resolucao"]))
    if item["disciplina"] != "Matemática":
        achados.erro(nome, "disciplina_invalida", repr(item["disciplina"]))
    if item["leitura_enunciado"] not in ("Aluno", "Aplicador"):
        achados.erro(nome, "leitura_enunciado_invalida", repr(item["leitura_enunciado"]))
    if not isinstance(item["depende_de_imagem"], bool):
        achados.erro(nome, "depende_de_imagem_nao_booleano", repr(item["depende_de_imagem"]))
    if item["codigo_bncc"] != habilidade:
        achados.erro(nome, "codigo_bncc_divergente_do_arquivo", repr(item["codigo_bncc"]))
    if "codigo_item" in item:
        achados.erro(nome, "codigo_item_inventado",
                     "é gerado na importação; escolhê-lo à mão colide entre alunos")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item["revisado_em"])):
        achados.erro(nome, "revisado_em_fora_do_formato", repr(item["revisado_em"]))
    origem = item["origem"]
    if not isinstance(origem, dict) or not origem.get("documento") \
            or origem.get("questao_no_documento") is None:
        achados.erro(nome, "origem_incompleta", repr(origem))

    # --- alternativas e gabarito -------------------------------------------
    alternativas = item["alternativas"]
    if not isinstance(alternativas, dict) or set(alternativas) != set(LETRAS):
        achados.erro(nome, "alternativas_devem_ser_A_a_D",
                     repr(sorted(alternativas) if isinstance(alternativas, dict) else alternativas))
        return None
    textos = [str(alternativas[l]).strip() for l in LETRAS]
    if not all(textos):
        achados.erro(nome, "alternativa_vazia")
    if len({t.upper() for t in textos}) < 4:
        achados.erro(nome, "alternativas_repetidas", repr(textos))
    gabarito = item["gabarito"]
    if gabarito not in LETRAS:
        achados.erro(nome, "gabarito_invalido", repr(gabarito))
        return None

    # --- justificativas ----------------------------------------------------
    justificativas = item["justificativas"]
    if not isinstance(justificativas, dict) or set(justificativas) != set(LETRAS):
        achados.erro(nome, "justificativas_devem_ser_A_a_D", repr(justificativas))
        return None
    for letra in LETRAS:
        if not str(justificativas[letra]).strip():
            achados.erro(nome, "justificativa_vazia", letra)
    if not str(item["justificativa_geral"]).strip():
        achados.erro(nome, "justificativa_geral_vazia")

    just_correta = str(justificativas[gabarito])
    if not just_correta.startswith("Correto - "):
        achados.erro(nome, "justificativa_correta_sem_prefixo", repr(just_correta[:60]))
    contas = EXPR.findall(normalize_math(just_correta))
    if len(contas) > 1:
        achados.erro(nome, "mais_de_uma_conta_na_justificativa",
                     f"{len(contas)} contas; o verificador lê só a primeira")
    elif not contas:
        achados.aviso(nome, "nao_verificavel_sem_conta", repr(just_correta[:70]))

    # --- operadores ASCII ---------------------------------------------------
    achados_unicode = sorted(UNICODE_OPS & set(json.dumps(item, ensure_ascii=False)))
    if achados_unicode:
        achados.erro(nome, "operador_tipografico", "".join(achados_unicode))

    # --- imagem -------------------------------------------------------------
    enunciado_completo = f"{item['enunciado_item']} {item.get('texto_auxiliar') or ''}"
    if item["depende_de_imagem"] is False:
        achado = IMAGE_PATTERN.search(enunciado_completo)
        if achado:
            achados.erro(nome, "autocontida_mas_cita_figura", achado.group())
    imagem = item["imagem"]
    if imagem:
        if not str(imagem).startswith(f"{habilidade}/"):
            achados.erro(nome, "imagem_fora_da_pasta_da_habilidade", str(imagem))
        elif not (dir_figuras / imagem).exists():
            achados.erro(nome, "imagem_inexistente", str(imagem))
        if not str(item["descricao_imagem"] or "").strip():
            achados.erro(nome, "descricao_imagem_ausente")
    elif item["depende_de_imagem"] is True:
        achados.erro(nome, "depende_de_imagem_mas_sem_arquivo")
    alts_img = item["alternativas_imagem"]
    if not isinstance(alts_img, dict) or set(alts_img) != set(LETRAS):
        achados.erro(nome, "alternativas_imagem_devem_ser_A_a_D", repr(alts_img))

    # --- contrato do app: o item precisa sobreviver ao verificador real -----
    questao = _questao_no_contrato(item)
    flags = check_structure({"questoes": [questao]}, quantidade_esperada=1)
    reprovadas = [k for k, v in flags.items() if not v]
    if reprovadas:
        achados.erro(nome, "reprova_check_structure", ",".join(reprovadas))
    tamanho = len(json.dumps({"questoes": [questao]}, ensure_ascii=False))
    if tamanho > MAX_CHARS_RESPOSTA:
        achados.erro(nome, "resposta_montada_acima_do_limite",
                     f"{tamanho} > {MAX_CHARS_RESPOSTA} chars")
    consistente, sugestao = check_consistency(questao)
    if consistente is False:
        achados.erro(nome, "gabarito_nao_bate_com_a_conta",
                     f"a conta aponta para {sugestao or 'nenhuma alternativa'}")

    # --- qualidade pedagógica ----------------------------------------------
    # Regra 3 do guia: a alternativa correta começa pelo número da resposta.
    if not NUM_INICIAL.match(normalize_math(str(alternativas[gabarito]).strip())):
        achados.aviso(nome, "alternativa_correta_nao_inicia_por_numero",
                      repr(str(alternativas[gabarito])[:50]))
    iniciais = []
    for letra in LETRAS:
        achado = NUM_INICIAL.match(normalize_math(str(alternativas[letra]).strip()))
        iniciais.append(achado.group() if achado else None)
    if len(set(iniciais)) < 4 and None not in iniciais:
        achados.aviso(nome, "alternativas_com_mesmo_numero_inicial", repr(iniciais))

    # Pergunta de identificação respondida com quantidade: a resposta correta
    # não responde à pergunta feita, e a resolução (uma decomposição do próprio
    # número) não executa a comparação pedida. Treinar nisso ensina o modelo a
    # emitir `resolucao_passo_a_passo` desacoplada de `resposta_correta`.
    pergunta = str(item.get("texto_auxiliar") or item["enunciado_item"])
    if (PERGUNTA_IDENTIFICACAO.search(_deacc(pergunta))
            and not PERGUNTA_QUANTIDADE.search(_deacc(pergunta))
            and NUM_INICIAL.match(normalize_math(str(alternativas[gabarito]).strip()))):
        achados.erro(nome, "pergunta_de_identificacao_respondida_com_quantidade",
                     f"{pergunta.strip()[:60]!r} -> {str(alternativas[gabarito])[:30]!r}")

    return item


def valida_entrega(raiz, dir_json=None, dir_figuras=None):
    raiz = Path(raiz)
    dir_json = Path(dir_json) if dir_json else (raiz / "JSON" if (raiz / "JSON").is_dir() else raiz)
    dir_figuras = Path(dir_figuras) if dir_figuras else (
        raiz / "Figuras" if (raiz / "Figuras").is_dir() else dir_json
    )

    achados = Achados()
    itens, por_arquivo = [], {}
    arquivos = sorted(p for p in dir_json.glob("*.json") if p.name != "manifesto.json")
    if not arquivos:
        raise SystemExit(f"Nenhum .json de questões em {dir_json}")

    for caminho in arquivos:
        habilidade = caminho.stem
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            achados.erro(caminho.name, "json_invalido", str(exc))
            continue
        if not isinstance(dados, list):
            achados.erro(caminho.name, "json_nao_e_lista", type(dados).__name__)
            continue

        vistos = Counter()
        aceitos = []
        for bruto in dados:
            nome = f"{habilidade}#{bruto.get('id_local')}"
            vistos[bruto.get("id_local")] += 1
            item = valida_item(bruto, nome, habilidade, dir_figuras, achados)
            if item is not None:
                aceitos.append((nome, item))
        for id_local, vezes in vistos.items():
            if vezes > 1:
                achados.erro(f"{habilidade}#{id_local}", "id_local_duplicado", f"{vezes}x")

        # Equilíbrio de gabarito por arquivo (guia: nenhuma letra acima de 35%).
        gabaritos = Counter(i["gabarito"] for _, i in aceitos)
        if aceitos:
            letra, vezes = gabaritos.most_common(1)[0]
            if vezes / len(aceitos) > MAX_FRACAO_GABARITO:
                achados.aviso(caminho.name, "gabarito_desequilibrado",
                              f"{letra} em {100 * vezes / len(aceitos):.0f}% "
                              f"(limite {100 * MAX_FRACAO_GABARITO:.0f}%)")
        por_arquivo[habilidade] = aceitos
        itens.extend(aceitos)

    # --- duplicidade dentro da entrega -------------------------------------
    assinaturas = defaultdict(list)
    enunciados = defaultdict(list)
    for nome, item in itens:
        completo = f"{item['enunciado_item']} {item.get('texto_auxiliar') or ''}"
        alternativas = "|".join(str(item["alternativas"][l]) for l in LETRAS)
        assinaturas[_norm_chave(completo + "||" + alternativas)].append(nome)
        enunciados[_norm_chave(completo)].append(nome)
    for nomes in assinaturas.values():
        if len(nomes) > 1:
            for nome in nomes[1:]:
                achados.erro(nome, "questao_duplicada_na_entrega", f"igual a {nomes[0]}")
    for nomes in enunciados.values():
        if len(nomes) > 1:
            for nome in nomes[1:]:
                achados.aviso(nome, "enunciado_repetido_na_entrega", f"igual a {nomes[0]}")

    # --- imagens órfãs ------------------------------------------------------
    referenciadas = {str(i["imagem"]) for _, i in itens if i["imagem"]}
    if dir_figuras.is_dir():
        em_disco = {str(p.relative_to(dir_figuras)) for p in dir_figuras.rglob("*.png")}
        for orfa in sorted(em_disco - referenciadas):
            achados.aviso(orfa, "imagem_orfa", "no disco, nenhuma questão referencia")

    return itens, achados, por_arquivo


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrega", help="pasta da entrega (com JSON/ e Figuras/)")
    parser.add_argument("--json", help="grava o relatório detalhado neste arquivo")
    parser.add_argument("--quiet", action="store_true", help="só o resumo")
    args = parser.parse_args()

    itens, achados, por_arquivo = valida_entrega(args.entrega)
    com_erro = set(achados.erros)
    aprovados = [n for n, _ in itens if n not in com_erro]
    com_alerta = [n for n in aprovados if n in achados.avisos]

    if not args.quiet:
        for titulo, tabela in (("ERROS", achados.erros), ("AVISOS", achados.avisos)):
            codigos = Counter(c for lista in tabela.values() for c, _ in lista)
            print(f"\n=== {titulo} ({sum(codigos.values())}) ===")
            for codigo, vezes in codigos.most_common():
                print(f"  {codigo:<48} {vezes:>4}")
                exemplos = [(n, d) for n, l in tabela.items() for c, d in l if c == codigo]
                for nome, detalhe in exemplos[:3]:
                    print(f"      · {nome}: {detalhe[:110]}")

    print(f"\n{len(por_arquivo)} arquivo(s), {len(itens)} item(ns) lidos")
    print(f"  APROVADOS .............. {len(aprovados) - len(com_alerta)}")
    print(f"  APROVADOS COM ALERTA ... {len(com_alerta)}")
    print(f"  REJEITADOS ............. {len(com_erro & {n for n, _ in itens}) + 0}")
    total_erros = sum(len(v) for v in achados.erros.values())
    print(f"\n{len(por_arquivo)} arquivo(s), {total_erros} erro(s), "
          f"{sum(len(v) for v in achados.avisos.values())} aviso(s)")
    print("Entrega válida." if total_erros == 0 else "Entrega INVÁLIDA — corrija os erros.")

    if args.json:
        relatorio = {
            "entrega": str(args.entrega),
            "itens_lidos": len(itens),
            "aprovados": sorted(set(aprovados) - set(com_alerta)),
            "aprovados_com_alerta": sorted(com_alerta),
            "rejeitados": {n: [{"codigo": c, "detalhe": d} for c, d in l]
                           for n, l in sorted(achados.erros.items())},
            "avisos": {n: [{"codigo": c, "detalhe": d} for c, d in l]
                       for n, l in sorted(achados.avisos.items())},
        }
        Path(args.json).write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Relatório detalhado: {args.json}")

    return 0 if total_erros == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
