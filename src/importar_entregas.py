"""Incorpora uma entrega validada em DB/questoes.db, com rastreabilidade.

É o passo `importar_entregas.py (professor)` do fluxo descrito no
GUIA_ENTREGA_ALUNOS.md, que não existia no repositório.

Três invariantes deste script:

1. NADA entra sem passar por validar_entrega.py. Item com qualquer ERRO é
   rejeitado e o motivo fica registrado no relatório — não há correção
   silenciosa de dado semanticamente errado.

2. O BLOB `imagem` só é preenchido quando `depende_de_imagem` é true.
   extract_data.build_example() descarta toda linha com coluna de imagem
   não-nula (o modelo em produção é texto puro). Gravar o PNG em todos os itens
   com figura repetiria exatamente a perda que o guia existe para evitar: das
   180 questões deste lote que têm PNG, só 10 dependem dele para ser resolvidas.
   As outras 170 foram textualizadas pelo autor e precisam chegar ao treino.
   A figura dessas 170 não é descartada — fica em `imagem_path`, preservada
   para um futuro modelo multimodal, exatamente como o guia promete ao aluno.

3. Colunas novas são ADITIVAS e nullable. As 553 linhas antigas continuam com
   NULL nelas e o comportamento de extract_data.py sobre elas é bit a bit o
   mesmo de antes — nenhum consumidor de `SELECT *` quebra.

Reexecutar é seguro: as linhas do mesmo `--lote` são removidas antes da
inserção (idempotente). O banco anterior é sempre copiado para DB/ antes de
qualquer escrita.

Uso:
    python src/importar_entregas.py TreinamentoDados/ --lote L2-2026-09 --dry-run
    python src/importar_entregas.py TreinamentoDados/ --lote L2-2026-09
"""

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from validar_entrega import LETRAS, valida_entrega

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "DB" / "questoes.db"

# Colunas acrescentadas por este ciclo. Todas nullable: as linhas pré-existentes
# ficam com NULL e nada muda para elas.
COLUNAS_NOVAS = [
    ("codigo_bncc", "TEXT"),        # EF01MA02 — habilidade da BNCC, do documento
    ("depende_de_imagem", "INTEGER"),  # 1 = item só é resolvível vendo a figura
    ("imagem_path", "TEXT"),        # caminho do PNG preservado (não é BLOB: não filtra o treino)
    ("descricao_imagem", "TEXT"),
    ("origem_documento", "TEXT"),   # rastreabilidade até o arquivo de origem
    ("origem_questao", "INTEGER"),  # número da questão dentro daquele documento
    ("autor", "TEXT"),
    ("revisado_em", "TEXT"),
    ("lote", "TEXT"),               # identifica esta importação
]


def garante_colunas(con):
    existentes = {r[1] for r in con.execute("PRAGMA table_info(itens)")}
    criadas = []
    for nome, tipo in COLUNAS_NOVAS:
        if nome not in existentes:
            con.execute(f"ALTER TABLE itens ADD COLUMN {nome} {tipo}")
            criadas.append(nome)
    return criadas


def monta_linha(item, lote, dir_figuras):
    """Traduz um item da entrega para uma linha da tabela `itens`."""
    depende = bool(item["depende_de_imagem"])
    imagem_rel = item["imagem"]

    # Regra 2 do cabeçalho: BLOB só para quem depende da figura.
    blob = None
    if depende and imagem_rel:
        blob = (dir_figuras / imagem_rel).read_bytes()

    justificativas = item["justificativas"]
    return {
        # `habilidade` recebe o código BNCC: o banco só tem matriz SAEB (H01-H26)
        # para 2º/5º/9º, e este lote traz 1º/3º/4º. Inventar um "H01 de 1º ano"
        # fabricaria uma informação que não existe em nenhuma matriz oficial.
        "codigo_item": f"{item['codigo_bncc']}-{item['id_local']:03d}-{lote}",
        "enunciado_item": item["enunciado_item"],
        "texto_auxiliar": item.get("texto_auxiliar"),
        "alternativa_a": item["alternativas"]["A"],
        "alternativa_b": item["alternativas"]["B"],
        "alternativa_c": item["alternativas"]["C"],
        "alternativa_d": item["alternativas"]["D"],
        "gabarito": item["gabarito"],
        "imagem": blob,
        "tipo": item["tipo"],
        "grau_resolucao": item["grau_resolucao"],
        "leitura_enunciado": item["leitura_enunciado"],
        "justificativa_geral": item["justificativa_geral"],
        "justificativa_alternativa_a": justificativas["A"],
        "justificativa_alternativa_b": justificativas["B"],
        "justificativa_alternativa_c": justificativas["C"],
        "justificativa_alternativa_d": justificativas["D"],
        "ano": item["ano"],
        "habilidade": item["codigo_bncc"],
        "disciplina": item["disciplina"],
        "descricao_item": item["descricao_item"],
        "codigo_bncc": item["codigo_bncc"],
        "depende_de_imagem": int(depende),
        "imagem_path": imagem_rel,
        "descricao_imagem": item.get("descricao_imagem"),
        "origem_documento": item["origem"]["documento"],
        "origem_questao": item["origem"]["questao_no_documento"],
        "autor": item["autor"],
        "revisado_em": item["revisado_em"],
        "lote": lote,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrega", help="pasta da entrega (com JSON/ e Figuras/)")
    parser.add_argument("--lote", required=True, help="identificador do lote, ex: L2-2026-09")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="não escreve nada")
    parser.add_argument("--relatorio", help="grava o relatório de incorporação neste .json")
    parser.add_argument("--incluir-alertas", action="store_true", default=True,
                        help="incorpora também os APROVADOS COM ALERTA (padrão)")
    args = parser.parse_args()

    entrega = Path(args.entrega)
    dir_figuras = entrega / "Figuras" if (entrega / "Figuras").is_dir() else entrega / "JSON"

    itens, achados, _ = valida_entrega(entrega)
    rejeitados = set(achados.erros)
    aprovados = [(n, i) for n, i in itens if n not in rejeitados]
    com_alerta = {n for n, _ in aprovados if n in achados.avisos}

    a_incorporar = aprovados if args.incluir_alertas else [
        (n, i) for n, i in aprovados if n not in com_alerta]

    print(f"Entrega:      {entrega}")
    print(f"Lote:         {args.lote}")
    print(f"Itens lidos:  {len(itens)}")
    print(f"  APROVADOS ............ {len(aprovados) - len(com_alerta)}")
    print(f"  APROVADOS C/ ALERTA .. {len(com_alerta)}")
    print(f"  REJEITADOS ........... {len(rejeitados & {n for n, _ in itens})}")
    motivos = Counter(c for nome, lista in achados.erros.items() for c, _ in lista)
    for codigo, vezes in motivos.most_common():
        print(f"      {codigo}: {vezes}")

    textuais = sum(1 for _, i in a_incorporar if not i["depende_de_imagem"])
    print(f"\nA incorporar: {len(a_incorporar)}"
          f"  ({textuais} chegam ao treino de texto, "
          f"{len(a_incorporar) - textuais} ficam só no banco por dependerem da figura)")

    relatorio = {
        "lote": args.lote,
        "entrega": str(entrega),
        "importado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "itens_lidos": len(itens),
        "incorporados": len(a_incorporar),
        "incorporados_treino_texto": textuais,
        "incorporados_dependentes_de_imagem": len(a_incorporar) - textuais,
        "aprovados_com_alerta": sorted(com_alerta),
        "rejeitados": {n: [{"codigo": c, "detalhe": d} for c, d in l]
                       for n, l in sorted(achados.erros.items())},
        "avisos": {n: [{"codigo": c, "detalhe": d} for c, d in l]
                   for n, l in sorted(achados.avisos.items())},
        "distribuicao_incorporada": {
            "ano": dict(Counter(i["ano"] for _, i in a_incorporar)),
            "habilidade": dict(Counter(i["codigo_bncc"] for _, i in a_incorporar)),
            "grau_resolucao": dict(Counter(i["grau_resolucao"] for _, i in a_incorporar)),
            "gabarito": dict(Counter(i["gabarito"] for _, i in a_incorporar)),
        },
    }

    if args.dry_run:
        print("\n[dry-run] nada foi escrito.")
    else:
        db = Path(args.db)
        carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = db.with_name(f"{db.stem}.antes-de-{args.lote}-{carimbo}.db")
        shutil.copy2(db, backup)
        print(f"\nBackup do banco: {backup}")
        relatorio["backup_db"] = str(backup)

        con = sqlite3.connect(db)
        criadas = garante_colunas(con)
        if criadas:
            print(f"Colunas acrescentadas (aditivas, nullable): {', '.join(criadas)}")
        relatorio["colunas_acrescentadas"] = criadas

        removidas = con.execute("DELETE FROM itens WHERE lote = ?", (args.lote,)).rowcount
        if removidas:
            print(f"Reimportação: {removidas} linha(s) do lote anterior removidas")

        linhas = [monta_linha(i, args.lote, dir_figuras) for _, i in a_incorporar]
        if linhas:
            campos = list(linhas[0])
            sql = (f"INSERT INTO itens ({', '.join(campos)}) "
                   f"VALUES ({', '.join('?' for _ in campos)})")
            con.executemany(sql, [[l[c] for c in campos] for l in linhas])
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM itens").fetchone()[0]
        do_lote = con.execute("SELECT COUNT(*) FROM itens WHERE lote = ?",
                              (args.lote,)).fetchone()[0]
        con.close()
        print(f"Inseridas: {do_lote}   |   Banco agora: {total} linhas")
        relatorio["linhas_no_banco_apos"] = total

    if args.relatorio:
        Path(args.relatorio).write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Relatório: {args.relatorio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
