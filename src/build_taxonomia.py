"""
Gera data/taxonomia_subtemas.json: habilidade -> subtemas -> tipos de raciocínio,
mais listas GLOBAIS de contextos narrativos e estruturas de enunciado.

Por que existe: o único eixo de variação do professor/inferência era o contexto
narrativo sorteado (TEMAS). Contexto não é diversidade matemática. Esta taxonomia
dá ao planejador (src/diversidade.py) um eixo de SUBTEMA matemático derivado do
próprio descritor da habilidade.

Derivação GENÉRICA (nenhuma regra por código H — o mesmo H muda de sentido por
ano, então a chave é sempre (ano, habilidade) e a fonte é descricao_item):
  (a) decomposição da descrição: enumerações entre parênteses separadas por
      vírgula/"ou"/"e" (ex.: 9º H16 lista condição de existência, soma dos
      ângulos internos...). Cada item vira subtema; se o item casa com o léxico,
      usa-se a entrada do léxico (tem palavras-chave melhores).
  (b) léxico determinístico de objetos matemáticos aplicado à descrição inteira;
      só entra o que CASA com a descrição (não inventa). Termos genéricos
      ("figuras planas") expandem para seus hipônimos, que por definição estão
      contidos no descritor; restrições textuais ("malha quadriculada") excluem
      hipônimos incompatíveis (círculo).
  (c) validação contra o banco: conta quantos itens reais caem em cada subtema
      (n_exemplos_db) usando o mesmo classificador de src/diversidade.py.

Override manual opcional: data/taxonomia_overrides.json no formato
  {"9º|H17": {"remover": ["id"], "adicionar": [{subtema...}], "tipos_raciocinio": [...]}}

Todas as regex estão em texto NORMALIZADO (minúsculas, sem acento) — ver
diversidade.normalizar_texto.

Uso: venv/bin/python src/build_taxonomia.py [--db DB/questoes.db] [--out data/taxonomia_subtemas.json]
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from diversidade import normalizar_texto, CONTEXTOS, ESTRUTURAS, TIPOS_RACIOCINIO  # noqa: E402

DB_PATH = ROOT / "DB" / "questoes.db"
OUT_PATH = ROOT / "data" / "taxonomia_subtemas.json"
OVERRIDES_PATH = ROOT / "data" / "taxonomia_overrides.json"

# --------------------------------------------------------------------------
# Léxico de objetos matemáticos: (id, rótulo, regex_na_descricao, palavras_chave_questao)
# A regex na descrição decide SE o subtema existe; palavras_chave classificam questões.
# --------------------------------------------------------------------------
LEXICO = [
    # geometria plana
    ("triangulo", "triângulos", r"triangul", r"triangul|isosceles|escaleno|equilater|acutangul|obtusangul"),
    ("quadrilatero", "quadriláteros", r"quadrilater",
     r"quadrilater|quadrado|(?<!triangulo )retangul|losango|trapezio|paralelogramo"),
    ("paralelogramo", "paralelogramos", r"paralelogram", r"paralelogram"),
    ("trapezio", "trapézios", r"trapezi", r"trapezi"),
    ("losango", "losangos", r"losang", r"losang"),
    ("retangulo_quadrado", "retângulos e quadrados", r"(?<!triangulo )retangulo(?!s? ret)|quadrado(?! magico)",
     r"(?<!triangulo )retangul|quadrado|comprimento e .{0,30}largura"),
    ("circulo", "círculo e circunferência", r"circul|circunferen", r"circul|circunferen|\braio\b|diametro"),
    ("poligono", "polígonos", r"poligon", r"poligon|pentagon|hexagon|octogon"),
    # elementos da circunferência
    ("raio_diametro", "raio e diâmetro", r"\braio\b|diametro", r"\braio\b|diametro"),
    ("corda", "corda", r"\bcorda\b", r"\bcorda"),
    ("arco", "arco", r"\barco\b", r"\barco"),
    ("angulo_central_inscrito", "ângulo central e ângulo inscrito", r"angulo central|angulo inscrito",
     r"angulo (central|inscrito)"),
    ("comprimento_circunferencia", "comprimento da circunferência", r"comprimento da circunferencia",
     r"comprimento da circunferencia|contorno|volta"),
    # propriedades de triângulos
    ("condicao_existencia", "condição de existência do triângulo", r"condicao de existencia",
     r"exist|possivel (formar|construir|montar)|desigualdade triangular|nao (e possivel|forma)|possa(m)? ser (construid|formad|montad)|pode(m)? (formar|construir|montar)"),
    ("relacao_lados_angulos", "relação entre lados e ângulos opostos", r"relac(ao|oes) de ordem|relac\w* entre .{0,40}lados",
     r"maior lado|menor lado|oposto|maior angulo|menor angulo"),
    ("soma_angulos_internos", "soma dos ângulos internos", r"soma dos angulos internos",
     r"soma dos angulos|180|terceiro angulo"),
    ("angulo_interno_medida", "medida de ângulo interno", r"medida de um angulo interno",
     r"medida do angulo|quanto mede|valor de x|angulo \w+ mede"),
    ("angulo_externo", "ângulo externo", r"angulo (interno ou )?externo", r"extern"),
    # ângulos / retas
    ("angulos_poligonos", "ângulos em polígonos", r"angulos em poligon", r"angul.*poligon|poligon.*angul"),
    ("retas_paralelas", "retas paralelas e transversal", r"retas paralelas", r"paralel|transversal"),
    ("cevianas", "elementos notáveis (cevianas)", r"ceviana", r"mediana|bissetriz|altura|ceviana"),
    # geometria espacial
    ("prisma", "prismas e blocos retangulares", r"prisma|bloco retangular|paralelepiped|\bcubo",
     r"prisma|bloco|paralelepiped|\bcubo|caixa"),
    ("piramide", "pirâmides", r"piramide", r"piramide"),
    ("cilindro", "cilindros", r"cilindr", r"cilindr|lata|tambor|cano"),
    ("cone", "cones", r"\bcone", r"\bcone"),
    ("esfera", "esferas", r"esfera", r"esfera|bola"),
    ("planificacao", "planificações", r"planifica", r"planifica|molde"),
    ("vistas", "vistas", r"\bvistas?\b", r"vista (frontal|lateral|superior|de cima)"),
    # grandezas e medidas
    ("comprimento", "medidas de comprimento", r"comprimento", r"\d\s*(km|cm|mm|m)\b|metro|centimetro|comprimento|altura|distancia"),
    ("massa", "medidas de massa", r"massa", r"\d\s*(kg|g|mg|t)\b|quilo|grama|massa|pes[ao]"),
    ("capacidade", "medidas de capacidade", r"capacidade", r"\d\s*(l|ml)\b|litro|mililitro|capacidade"),
    ("tempo_medida", "medidas de tempo", r"\btempo\b", r"hora|minuto|segundo|\bdias?\b|semana"),
    ("temperatura", "temperatura", r"temperatura", r"temperatura|grau"),
    ("volume_medida", "volume", r"\bvolume\b", r"volume|cm3|m3|cubic"),
    ("perimetro", "perímetro", r"perimetro", r"perimetro|contorno|cerca|moldura|volta"),
    ("area", "área", r"\barea\b", r"\barea\b|m2|cm2|metros quadrados|piso|azulejo|quadradinhos|malha quadriculada|comprimento e .{0,30}largura"),
    # tempo / calendário
    ("relogio_analogico", "horas em relógio analógico", r"analogic", r"analogic|ponteiro"),
    ("relogio_digital", "horas em relógio digital", r"digita", r"digital|\d{1,2}:\d{2}|\d{1,2}h\d{2}"),
    ("horario_inicio", "horário de início", r"inicio", r"comec|inici|saiu|partiu"),
    ("horario_termino", "horário de término", r"termino", r"termin|acab|chegou|fim"),
    ("duracao", "duração", r"duracao", r"dur(ou|a|acao)|quanto tempo|levou"),
    ("datas", "datas", r"\bdatas?\b", r"\bdata\b|/\d{2}/|\bdia \d"),
    ("dias_semana", "dias da semana", r"dias da semana", r"segunda|terca|quarta|quinta|sexta|sabado|domingo|dia da semana"),
    ("meses", "meses do ano", r"meses", r"janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|\bmes\b"),
    ("calendario", "calendário", r"calendario", r"calendario"),
    # estatística
    ("tabela_simples", "tabela simples", r"tabela", r"tabela"),
    ("tabela_dupla", "tabela de dupla entrada", r"dupla entrada", r"dupla entrada|linhas? e colunas?"),
    ("grafico_barras", "gráfico de barras", r"barras", r"barra"),
    ("grafico_colunas", "gráfico de colunas", r"colunas", r"coluna"),
    ("grafico_pictorico", "gráfico pictórico", r"pictoricos?\b", r"pictoric|cada (simbolo|figura|desenho) (vale|representa)"),
    ("grafico_linhas", "gráfico de linhas", r"de linhas", r"grafico de linha|linha"),
    ("grafico_setores", "gráfico de setores", r"setores", r"setor|pizza"),
    ("histograma", "histograma", r"histograma", r"histograma|intervalo|classe"),
    ("listas", "listas", r"\blistas\b", r"lista"),
    ("media", "média aritmética", r"\bmedia\b", r"\bmedia\b"),
    ("moda", "moda", r"\bmoda\b", r"\bmoda\b|mais frequente"),
    ("mediana", "mediana", r"mediana", r"mediana|valor central"),
    # probabilidade
    ("chance_maior_menor", "eventos com maior/menor chance", r"(maior|menor).{0,30}chances", r"mais (chance|provavel)|menos (chance|provavel)|maior chance|menor chance|qual (e )?a chance|chance de"),
    ("chance_igual", "eventos com chances iguais", r"iguais chances", r"mesma chance|iguais chances|igualmente|equiprovav"),
    ("eventos_independentes", "eventos independentes", r"independentes", r"independen|com reposicao|lanca.*e.*lanca"),
    ("eventos_dependentes", "eventos dependentes", r"(?<!in)dependentes", r"(?<!in)dependen|sem reposicao|retira.*depois"),
    # números e operações
    ("adicao", "adição", r"adic", r"adic|som[ae]|\+|mais|total|junt"),
    ("subtracao", "subtração", r"subtra", r"subtra|diferen|\-|menos|rest|sobr"),
    ("multiplicacao", "multiplicação", r"multiplica", r"multiplic|vezes|produto|\bx\b|×"),
    ("divisao", "divisão", r"divis(ao|oes)", r"divi|reparti|÷|cada um|igualmente"),
    ("potenciacao", "potenciação", r"potenciacao", r"potencia|\^|ao quadrado|ao cubo|expoente"),
    ("radiciacao", "radiciação", r"radicia", r"raiz|radic|√"),
    ("notacao_cientifica", "notação científica", r"notacao cientifica", r"notacao cientifica|x ?10\^|10 elevado"),
    ("fracao", "frações", r"fraciona|fraco|fracoes", r"frac|/\d|metade|terco|quarto|partes iguais"),
    ("decimal", "números decimais", r"decima", r"\d,\d|decima"),
    ("porcentagem", "porcentagem", r"porcent|percentua", r"%|porcent|por cento"),
    ("acrescimo", "acréscimos", r"acrescimo", r"acrescim|aument|reajust|juros"),
    ("decrescimo", "decréscimos (descontos)", r"decrescimo", r"desconto|reduc|diminu|decresc"),
    ("taxas_sucessivas", "percentuais sucessivos", r"(taxas|percentuais) sucessiv", r"sucessiv|depois .*%|em seguida.*%"),
    ("taxa_percentual", "determinação de taxa percentual", r"taxas? percentua|determinacao das taxas", r"qual (foi )?(a )?(taxa|porcentagem|percentual)"),
    ("multiplo", "múltiplos", r"multiplo", r"multiplo"),
    ("divisor", "divisores", r"divisor", r"divisor|divisivel"),
    ("mdc", "máximo divisor comum", r"mdc|maximo divisor", r"mdc|maior divisor|maximo divisor|maior numero possivel|mesmo tamanho"),
    ("mmc", "mínimo múltiplo comum", r"mmc|minimo multiplo", r"mmc|menor multiplo|minimo multiplo|novamente juntos|ao mesmo tempo"),
    ("equacao", "equações de 1º grau", r"equac", r"equac|=|incognita|\bx\b"),
    ("inequacao", "inequações de 1º grau", r"inequac", r"inequac|<|>|no minimo|no maximo|pelo menos"),
    ("sistema", "sistemas de equações", r"sistemas? de equac", r"sistema|duas incognitas|\bx\b.*\by\b"),
    # significados das operações (BNCC/SAEB)
    ("sig_juntar", "significado de juntar", r"juntar", r"junt|ao todo|total"),
    ("sig_acrescentar", "significado de acrescentar", r"acrescentar", r"acrescent|ganhou|mais \d|chegaram"),
    ("sig_separar", "significado de separar", r"separar", r"separ"),
    ("sig_retirar", "significado de retirar", r"retirar", r"retir|tirou|perdeu|gastou|deu \d|comeu|vendeu"),
    ("sig_comparar", "significado de comparar", r"significados? de .*comparar", r"a mais que|a menos que|diferenca|quantos? a mais|quantos? a menos"),
    ("sig_completar", "significado de completar", r"completar", r"falta|completar|faltam"),
    ("grupos_iguais", "grupos iguais / repartição", r"grupos iguais|reparticao", r"grupo|cada|reparti|igualmente"),
    ("proporcionalidade", "proporcionalidade", r"proporcional", r"proporc|dobro|triplo|cada \d"),
    ("disposicao_retangular", "disposição retangular", r"disposicao retangular", r"fileira|linhas? e colunas?|disposicao"),
    # sistema monetário / sequências / localização
    ("moedas", "moedas", r"moedas", r"moeda|centavo"),
    ("cedulas", "cédulas", r"cedulas", r"cedula|nota de|notas de"),
    ("seq_numerica", "sequências numéricas", r"sequencia.*numer|numeros naturais ordenados|sequencia recursiva",
     r"sequencia|\d+\s*,\s*\d+\s*,\s*\d+"),
    ("seq_figuras", "sequências de figuras/objetos", r"sequencia.*(figuras|objetos)", r"figura|objeto|desenho"),
    ("mapas_croquis", "mapas e croquis", r"mapa|croqui", r"mapa|croqui|rua|quarteirao"),
    ("plantas", "plantas de ambientes", r"plantas", r"planta"),
    ("localizacao", "localização", r"localizac", r"localiz|onde (esta|fica)|entre|ao lado|a direita|a esquerda"),
    ("deslocamento", "deslocamento", r"deslocamento", r"desloc|caminho|trajeto|vire|siga|andou"),
    # registro de números
    ("registro_algarismos", "registro por algarismos", r"representacao por algarismos|registro numerico", r"algarismo|\d"),
    ("registro_lingua_materna", "registro por extenso", r"lingua materna", r"por extenso|escrito|escreve|le-se"),
    ("valor_posicional", "valor posicional", r"valor posicional|ordem ocupada", r"ordem|posicional|unidade|dezena|centena|milhar"),
    ("reta_numerica", "reta numérica", r"reta numerica", r"reta"),
]

# Termos genéricos que o descritor usa para uma CLASSE de objetos. Expandir não é
# inventar: os hipônimos estão contidos no termo. `exclui_se` corta hipônimos
# incompatíveis com outra restrição textual do mesmo descritor.
HIPERONIMOS = [
    (r"figuras? (geometricas )?planas",
     ["retangulo_quadrado", "triangulo", "trapezio", "circulo", "poligono"],
     {"circulo": r"malha quadriculada", "poligono": r"malha quadriculada"}),
    # "polígonos" sozinho (sem "ângulos em polígonos", que é outro subtema)
    (r"(?<!angulos em )\bpoligonos\b", ["retangulo_quadrado", "triangulo", "trapezio", "poligono"], {}),
    (r"propriedades dos triangulos", ["triangulo"], {}),
    (r"figuras geometricas espaciais|objetos tridimensionais", ["prisma", "piramide", "cilindro", "cone", "esfera"], {}),
    (r"medidas de grandezas|conversao de unidades|medidas e conversao", ["comprimento", "massa", "capacidade", "tempo_medida"], {}),
    (r"tabelas e graficos", ["tabela_simples", "grafico_barras", "grafico_colunas"], {}),
    (r"tendencia central", ["media", "moda", "mediana"], {}),
    (r"elementos da circunferencia", ["raio_diametro", "corda", "arco", "angulo_central_inscrito"], {}),
    (r"numeros racionais positivos|representacoes? de (um )?numeros? racional", ["fracao", "decimal", "porcentagem"], {}),
]

# Quando uma entrada mais específica está presente, a genérica sobra: evita que
# "triângulos" e "propriedades dos triângulos" disputem a mesma questão.
SUBSUMIDO_POR = {
    "tabela_simples": [],
    "retangulo_quadrado": ["quadrilatero"],
    "circulo": ["comprimento_circunferencia", "raio_diametro", "corda", "arco", "angulo_central_inscrito"],
    "valor_posicional": [],
}

# Objetos/grandezas que costumam ser o TEMA da habilidade (área, perímetro,
# volume, triângulo...) e não um recorte dela. Só são movidos para
# `objetos_portadores` quando há >=2 outros subtemas e vieram do léxico direto
# (não de enumeração nem de expansão de hiperônimo).
OBJETOS_PORTADORES = ("triangulo", "quadrilatero", "circulo", "poligono", "perimetro", "area", "volume_medida")

STOP = set("a o as os de da do das dos e ou em um uma uns umas com sem por para que se no na nos nas "
           "ao aos ate entre outros outras etc suas seus sua seu the".split())


def _dict_lexico():
    return {e[0]: e for e in LEXICO}


def _entrada_para_subtema(e, origem):
    return OrderedDict(id=e[0], rotulo=e[1], palavras_chave=[e[3]], origem=origem)


def _itens_enumeracao(desc_norm):
    """Itens de enumerações entre parênteses (>=2 itens). Ignora parênteses que
    não enumeram nada ('(ou valor relativo)', '(até 2 ordens)')."""
    itens = []
    for grupo in re.findall(r"\(([^()]*)\)", desc_norm):
        partes = [p.strip(" .;") for p in re.split(r",", grupo) if p.strip(" .;")]
        # último item costuma vir como "x ou y" / "x e y": separa só se curto
        expand = []
        for p in partes:
            sub = re.split(r"\s+(?:ou|e)\s+", p)
            if len(sub) > 1 and all(len(s.split()) <= 3 for s in sub):
                expand.extend(sub)
            else:
                expand.append(p)
        if len(expand) >= 2:
            itens.extend(expand)
    return itens


def _slug(txt):
    pal = [w for w in re.findall(r"[a-z0-9]+", txt) if w not in STOP]
    return "_".join(pal[:5])


def _palavras_chave_frase(txt):
    """Regex de classificação para um subtema vindo de frase: radicais (5 letras)
    das palavras de conteúdo; exige que a questão contenha pelo menos um."""
    pal = [w for w in re.findall(r"[a-z]+", txt) if w not in STOP and len(w) >= 4
           and w not in ("determinar", "medida", "medidas", "relacoes", "identificar")]
    radicais = sorted({w[:5] for w in pal})
    return r"|".join(radicais) if radicais else r"$^"


def derivar_subtemas(descricoes):
    """Aplica (a) enumeração e (b) léxico/hiperônimos a TODAS as descrições da
    (ano, habilidade) (o banco tem variantes textuais do mesmo descritor)."""
    lex = _dict_lexico()
    achados = OrderedDict()
    for desc in descricoes:
        d = normalizar_texto(desc)
        # negações/opcionais ("sem utilizar frações", "com ou sem suporte da reta")
        # não definem conteúdo: removidos antes de casar o léxico
        d = re.sub(r"\bsem (utilizar |usar )?[^,.;()]*", "", d)
        # (a) enumeração: item com entrada no léxico usa o léxico; senão vira frase
        for item in _itens_enumeracao(d):
            casou = [e for e in LEXICO if re.search(e[2], item)]
            if casou:
                for e in casou:
                    achados.setdefault(e[0], _entrada_para_subtema(e, "enumeracao"))
            elif len({w for w in item.split() if w not in STOP and len(w) >= 3}) >= 2:
                sid = _slug(item)
                achados.setdefault(sid, OrderedDict(
                    id=sid, rotulo=item, palavras_chave=[_palavras_chave_frase(item)], origem="enumeracao"))
        # (b) léxico sobre o descritor inteiro
        for e in LEXICO:
            if re.search(e[2], d):
                achados.setdefault(e[0], _entrada_para_subtema(e, "lexico"))
        for rx, ids, exclui in HIPERONIMOS:
            if re.search(rx, d):
                for i in ids:
                    if i in exclui and re.search(exclui[i], d):
                        continue
                    achados.setdefault(i, _entrada_para_subtema(lex[i], "hiperonimo"))
    # remove genéricos subsumidos por específicos presentes
    for geral, especificos in SUBSUMIDO_POR.items():
        if geral in achados and any(s in achados for s in especificos):
            del achados[geral]
    # Objeto "portador": quando o descritor enumera propriedades/elementos de um
    # objeto ("elementos de um triângulo (condição..., soma...)"), o objeto é o
    # suporte comum de todos os subtemas e não um subtema concorrente — se ficasse,
    # absorveria todas as questões na classificação.
    objetos = []
    restantes = [i for i, v in achados.items() if i not in OBJETOS_PORTADORES or v["origem"] != "lexico"]
    if len(restantes) >= 2:
        for oid in OBJETOS_PORTADORES:
            if oid in achados and achados[oid]["origem"] == "lexico":
                objetos.append(achados.pop(oid)["rotulo"])
    return list(achados.values()), objetos


def derivar_tipos(descricoes):
    """Tipos de raciocínio habilitados pelos verbos/critérios do descritor."""
    d = " ".join(normalizar_texto(x) for x in descricoes)
    tipos = [t["id"] for t in TIPOS_RACIOCINIO if re.search(t["habilita"], d)]
    if len(tipos) < 2:
        # garante ao menos dois eixos de raciocínio; ambos são genéricos a
        # qualquer habilidade (reconhecer e aplicar)
        for extra in ("identificacao", "resolucao_problema"):
            if extra not in tipos:
                tipos.append(extra)
    return tipos


def carregar_itens(db):
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(itens)")}
    campos = ["ano", "habilidade", "descricao_item", "grau_resolucao"]
    extras = [c for c in ("enunciado_item", "texto_auxiliar", "alternativa_a", "alternativa_b",
                          "alternativa_c", "alternativa_d") if c in cols]
    rows = con.execute(f"SELECT {', '.join(campos + extras)} FROM itens WHERE disciplina = 'Matemática'").fetchall()
    con.close()
    return [dict(zip(campos + extras, r)) for r in rows]


def construir(db=DB_PATH, overrides_path=OVERRIDES_PATH):
    import diversidade
    itens = carregar_itens(db)
    grupos = OrderedDict()
    for it in sorted(itens, key=lambda x: (x["ano"] or "", x["habilidade"] or "")):
        if not it["ano"] or not it["habilidade"]:
            continue
        grupos.setdefault((it["ano"], it["habilidade"]), []).append(it)

    overrides = {}
    if Path(overrides_path).exists():
        overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8") or "{}")

    habilidades = OrderedDict()
    for (ano, hab), its in grupos.items():
        descricoes = sorted({i["descricao_item"].strip() for i in its if i.get("descricao_item")})
        subtemas, objetos = derivar_subtemas(descricoes)
        tipos = derivar_tipos(descricoes)
        chave = f"{ano}|{hab}"
        ov = overrides.get(chave, {})
        if ov.get("remover"):
            subtemas = [s for s in subtemas if s["id"] not in ov["remover"]]
        for s in ov.get("adicionar", []):
            subtemas.append(OrderedDict(s, origem="override"))
        if ov.get("tipos_raciocinio"):
            tipos = list(ov["tipos_raciocinio"])
        if not subtemas:
            # K=1: a habilidade inteira é o subtema (o planejador diversifica os outros eixos)
            subtemas = [OrderedDict(id="geral", rotulo=normalizar_texto(descricoes[0]).rstrip(". "),
                                    palavras_chave=[r"."], origem="descricao")]
        for s in subtemas:
            s["tipos_raciocinio"] = list(tipos)
        entrada = OrderedDict(ano=ano, habilidade=hab, descricoes=descricoes,
                              objetos_portadores=objetos,
                              graus_resolucao=sorted({i["grau_resolucao"] for i in its if i.get("grau_resolucao")}),
                              tipos_raciocinio=tipos, subtemas=subtemas, n_itens_db=len(its))
        # (c) validação contra o banco
        tax_local = {"habilidades": {chave: entrada}}
        cont = Counter()
        for i in its:
            texto = " ".join(str(i.get(c) or "") for c in ("enunciado_item", "texto_auxiliar", "alternativa_a",
                                                            "alternativa_b", "alternativa_c", "alternativa_d"))
            cont[diversidade.classificar_questao(texto, ano, hab, taxonomia=tax_local)["subtema"]] += 1
        for s in subtemas:
            s["n_exemplos_db"] = cont.get(s["id"], 0)
        entrada["n_sem_subtema_db"] = cont.get("outros", 0)
        habilidades[chave] = entrada

    return OrderedDict(
        versao=1,
        fonte="DB/questoes.db (disciplina=Matemática)",
        nota="Regex em texto normalizado (minúsculas, sem acentos). Chave = 'ano|habilidade'.",
        contextos=CONTEXTOS, estruturas=ESTRUTURAS,
        tipos_raciocinio=[{k: v for k, v in t.items() if k != "habilita"} for t in TIPOS_RACIOCINIO],
        habilidades=habilidades,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--overrides", default=str(OVERRIDES_PATH))
    args = ap.parse_args()
    tax = construir(args.db, args.overrides)
    Path(args.out).write_text(json.dumps(tax, ensure_ascii=False, indent=1), encoding="utf-8")
    for chave, h in tax["habilidades"].items():
        print(f"{chave:14s} K={len(h['subtemas']):2d} sem_subtema={h['n_sem_subtema_db']}/{h['n_itens_db']} "
              + ", ".join(f"{s['id']}({s['n_exemplos_db']})" for s in h["subtemas"]))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
