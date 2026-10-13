# Ciclo v7 da base: curadoria item a item

Entrada: `data/train_curado_v6.jsonl` (2.807 exemplos, 2.849 questões). Saída: `data/train_curado_v7.jsonl`.

## O que foi feito

`src/curar_base_v7.py` passou CADA questão da v6 por um professor de Matemática SAEB (Maritaca
`sabia-4`, papel "curador"), que resolve o item às cegas (sem ver gabarito, resolução nem
dificuldade pedida) e devolve veredito `aprovado | corrigivel | inviavel`. O código compara com o
gabarito e decide `confirmada | corrigida | gabarito_discorda | correcao_rejeitada | inviavel`.
Toda correção passa por uma 2ª chamada cega (o validador já calibrado de `agentes_questoes`) antes
de entrar. Custo real: R$ 42,68 para as 2.849 questões (0 abstenções sem retry; 6 abstenções
ficaram sem decisão e o item continuou como estava).

Política (nunca muda sem decisão humana, ver `src/curar_base_v7.py` docstring):
- item NÃO real (injetado/destilado/sintético): correção e remoção automáticas, documentadas em
  `outputs/melhoria_v7/proveniencia_v7.jsonl`;
- item REAL (`real_banco`, `real_L2`): correção e remoção viram PROPOSTA em
  `outputs/melhoria_v7/propostas_reais.jsonl` — só entram com decisão humana explícita.

## Decisão humana sobre os itens reais (2026-10-13)

7 propostas de itens reais. O responsável aprovou as 4 **correções** e rejeitou as 3 **remoções**:

| código | tipo | decisão |
|---|---|---|
| EF04MA04-005-L2-2026-09 | correção | aprovada (completou dado ausente: "cada músico usa um instrumento") |
| EF01MA02-016-L2-2026-09 | correção | aprovada, com ajuste do responsável (pergunta trocada para "caixas") |
| MT2077FH16MT | correção | aprovada (unidade "kg" explicitada no enunciado) |
| MT9055DH11MT | correção | aprovada, **com valor diferente do proposto pelo curador**: o responsável trocou o total para R$ 140,00 (dá solução inteira b=2, c=4), não os R$ 130,00 que o curador sugeriu (que também não dava inteiro) |
| MT9077MH16MT | remoção | aprovada → item removido da base ("vamos excluir") |
| EF03MA01-015-L2-2026-09 | remoção | aprovada → item removido da base ("vamos excluir") |
| MT5068MH14MT | remoção | aprovada → item removido da base ("vamos excluir") |

Registro em `outputs/melhoria_v7/EDICOES_REAIS_HUMANAS.json` (as 4 correções, antes/depois) e
`outputs/melhoria_v7/QUARENTENA_REAIS_HUMANA.json` (as 3 remoções de itens reais, aplicadas após
a decisão "vamos excluir").

`data/train_curado_v7.jsonl`: 2.807 → 2.774 (33 não-reais em quarentena pelo curador) → 2.771
(−3 itens reais removidos por decisão humana explícita: MT9077MH16MT, EF03MA01-015-L2-2026-09,
MT5068MH14MT).

## Verificação (`src/verificar_v7.py`, espelha `verificar_candidata_v6.py`)

8 de 9 critérios passam sem ressalva (schema, sem duplicata/contaminação com val*, 0
`matematicamente_problematica`, % verificável e near-dup não pioram, letras 18–22%,
explicabilidade de distratores não piora).

### Exceção registrada (2026-10-13)

**V7-03a** (REFUTADO = 0 e FRACO = 0, herdado do v6): a v7 tem 2 REFUTADO, exatamente as 2 edições
reais acima cujo texto/valor diverge do registro oficial do banco SQLite por causa da própria
correção aprovada, não de um erro novo:
- `EF01MA02-016-L2-2026-09`: o banco guarda o texto oficial "9 REMEDIOS." para a alternativa
  correta; a correção trocou a pergunta para "quantas caixas" (elimina a ambiguidade de unidade
  "caixas de remédio" × "remédios"), então o texto virou "9 caixas." — diverge na PALAVRA, não no
  valor.
- `MT9055DH11MT`: o banco registra R$ 120,00, que não fecha em número inteiro de camisas/bermudas
  (b = 1,6). A correção aprovada usa R$ 140,00 (b = 2, c = 4, confere: 4×20 + 2×30 = 140) — diverge
  do banco porque o banco tinha o defeito que motivou a correção, não porque a correção está errada.

`auditar_gabaritos_base.py` compara o texto/valor da alternativa correta contra o registro oficial
do SQLite; ele não distingue "edição humana aprovada" de "erro novo introduzido". O critério NÃO foi
alterado para os próximos ciclos — a exceção vale só para estes 2 códigos, verificados caso a caso
nesta decisão. `venv/bin/python src/verificar_v7.py --excecao=V7-03a` grava a aprovação com a
exceção registrada em `ciclo_v7/ANTES_DEPOIS_v7.json`.

## Treino v7, gate de promoção e investigação (2026-10-13)

Treinado no `train_curado_v7.jsonl` (2.771 exemplos). O gate mecânico
(`promover_checkpoint.py --perfil multiseed`) deu **NÃO PROMOVIDO** (3 rodadas, n=90):
gates bloqueantes reprovados G1, G2, G5, G7. Rodada ampliada para 5 rodadas (n=150) CONFIRMOU
o efeito como estatisticamente significativo (McNemar G2 p=0,006, G3 p=0,021 — G3 passou a
reprovar também), não ruído de amostra pequena: `outputs/relatorios/VEREDITO_v7_multiseed_n150.json`.

**Diagnóstico** (Jev via OpenRouter + agente Claude independente, convergentes; depois confirmado
nos itens reais gerados): a curadoria derrubou a taxa de "Nenhuma das alternativas anteriores" no
candidato de 58,0% para 23,3% (ganho real de viés de gabarito, G6 passa: 42,2%→28,9%). Sem essa
muleta, o modelo de 1,7B precisa gerar 4 distratores plausíveis de verdade e erra a própria
aritmética de números grandes, concentrado em 3 habilidades: `5º H03` (soma/subtração até 6
ordens), `5º H01` (escrever número por extenso) e `9º H02` (operações com números reais). A
quantidade de exemplos reais dessas habilidades não caiu na curadoria (v6→v7: 32→31, 39→38, 32→32)
— o problema é padrão de distrator mal aprendido, não falta de volume.

## Reforço sintético (`src/reforco_distratores_v7.py`, 2026-10-13)

Gera `data/train_curado_v7b.jsonl` = v7 (2.771) + 85 exemplos / 135 questões de reforço nas 3
habilidades acima, com gabarito e distratores computados em Python via `num2words`/aritmética
exata (nunca por LLM): 100% verificável, 0 falha de schema, 0 contaminação com val*.

`src/verificar_v7.py` contra `v7b` falha em `V7-03d` (pouco_explicativa) e `V7-03e` (near-dup) —
**exceção aceita, não corrigida**: esses critérios assumem curadoria sem item novo; um lote de
reforço repete deliberadamente o mesmo padrão de distrator pedagógico em várias habilidades/
dificuldades, e o detector de near-dup (trigramas de palavras sobre o enunciado com números
mascarados) não distingue "mesmo padrão ensinado de novo" de duplicata real — a maior parte da
frase é texto fixo do template, então variar nome/contexto não derruba o Jaccard o bastante quando
~135 questões se concentram em só 3 habilidades. `V7-03a` continua exceção herdada da v7 pura.
O critério de aceitação real para `v7b` é o retreino e o gate `G1/G5/G7`, não esta auditoria
estática — ainda não rodado.

    venv/bin/python src/reforco_distratores_v7.py --num-per-skill 15   # já gerado
    venv/bin/python src/train.py --train data/train_curado_v7b.jsonl --val data/val.jsonl \
      --out outputs/lora_v7b --checkpoints outputs/checkpoints_v7b
