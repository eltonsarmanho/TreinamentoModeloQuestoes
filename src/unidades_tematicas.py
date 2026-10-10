"""Mapa (ano, habilidade) -> unidade temática (BNCC).

Serve ao recorte por unidade da amostra humana (amostra_humana.py): a validade
das questões é reportada por unidade temática, não só no agregado.

Unidades: Números, Álgebra, Geometria, Grandezas e medidas,
Probabilidade e estatística.

Fonte da classificação: a DESCRIÇÃO de cada descritor em
Doc/Habilidades_Sistema.md (conferida contra DB/questoes.db e
data/taxonomia_subtemas.json em 2026-10-10). Pontos que não são óbvios:
  * 2º H09/H10 (classificar por atributos; sequências) e 5º H07-H09
    (padrões/sequências) são Álgebra na BNCC (regularidades e padrões);
  * 2º H13 e 5º H10-H11 (localização/deslocamento em mapas) são Geometria;
  * 2º H15 / 5º H19 (sistema monetário), 2º H17-H18 (sequência do dia,
    calendário) e 5º H17-H18 (relógio, duração) são Grandezas e medidas;
  * 9º H20-H21 (perímetro, área) são Grandezas e medidas (não Geometria);
  * 9º H07-H09 (frações, equivalência, conversão fração/decimal/%) são Números;
  * os códigos EFxxMAxx do 1º ao 4º ano do corpus (EF01MA02, EF01MA06,
    EF01MA08, EF02MA03, EF02MA05, EF02MA06, EF03MA01, EF03MA03, EF03MA07,
    EF04MA04) tratam de contagem, comparação de quantidades, leitura de números
    e fatos/problemas das quatro operações: Números (convenção do projeto,
    confirmada pela descrição).
A cobertura das 79 habilidades de data/train_curado_v3.jsonl é testada em
tests/test_unidades_tematicas.py.
"""
import re

NUMEROS = "Números"
ALGEBRA = "Álgebra"
GEOMETRIA = "Geometria"
GRANDEZAS = "Grandezas e medidas"
ESTATISTICA = "Probabilidade e estatística"

UNIDADES = (NUMEROS, ALGEBRA, GEOMETRIA, GRANDEZAS, ESTATISTICA)

# ano -> [(unidade, primeiro, último)] sobre o número do descritor Hnn
_FAIXAS = {
    "2º": [(NUMEROS, 1, 8), (ALGEBRA, 9, 10), (GEOMETRIA, 11, 13),
           (GRANDEZAS, 14, 18), (ESTATISTICA, 19, 21)],
    "5º": [(NUMEROS, 1, 6), (ALGEBRA, 7, 9), (GEOMETRIA, 10, 14),
           (GRANDEZAS, 15, 19), (ESTATISTICA, 20, 22)],
    "9º": [(NUMEROS, 1, 9), (ALGEBRA, 10, 13), (GEOMETRIA, 14, 18),
           (GRANDEZAS, 19, 22), (ESTATISTICA, 23, 26)],
}

MAPA = {(ano, f"H{n:02d}"): unidade
        for ano, faixas in _FAIXAS.items()
        for unidade, a, b in faixas for n in range(a, b + 1)}

_BNCC = re.compile(r"^EF0[1-4]MA\d{2}$")


def _norm_ano(ano):
    """'5º', '5º ano', '5', 5 -> '5º'."""
    m = re.match(r"\s*(\d)", str(ano))
    return f"{m.group(1)}º" if m else str(ano).strip()


def unidade_de(ano, habilidade):
    """Unidade temática de (ano, habilidade); None se não mapeada."""
    hab = str(habilidade).strip().upper()
    if _BNCC.match(hab):
        return NUMEROS
    return MAPA.get((_norm_ano(ano), hab))
