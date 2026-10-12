# Ciclo v6 da base de treino: especificação e critérios de aceite

Entrada imutável: `data/train_curado_v5.jsonl` (sha256 `41b8d159…`; travado em `ciclo_v6/ENTRADAS_IMUTAVEIS.json`).
Saída final: `data/train_curado_v6.jsonl`. Só pode ser gerada depois das aprovações listadas em "Portões".

## Escopo

- **Dentro:**
  - auditoria determinística da v5;
  - reescrita de resoluções pouco explicativas (só a explicação);
  - injeção direcionada às lacunas;
  - remoção de duplicatas não reais;
  - montagem e manifesto da candidata.
- **Fora:**
  - treino, que só acontece depois da candidata aprovada;
  - mudança de critérios de promoção;
  - mudança nas validações congeladas;
  - troca de gabarito;
  - edição do banco SQLite.

## Requisitos e critérios de aceite

ID, requisito e verificação:

| ID | Requisito (EARS) | Verificação |
|---|---|---|
| V6-01 | O sistema SEMPRE manterá v5, v3, train, val, val_frozen e DB/questoes.db byte a byte. | `tests/test_entradas_imutaveis.py`; `src/manifesto_v6.py` sai com código 1 se algum mudar |
| V6-02 | QUANDO uma chamada à API falhar, estourar o tempo, vier truncada ou com JSON inválido, o sistema registrará abstenção e NÃO aprovará o exemplo. | `tests/test_ciclo_v6.py::TestChamadorSabia/TestReescrita`; `tests/test_juiz_truncado.py` |
| V6-03 | O sistema NÃO enviará uma chamada que ultrapasse o teto de chamadas ou o teto em R$ (pior caso reservado antes do envio). | `TestChamadorSabia::test_teto_*` |
| V6-04 | ENQUANTO estiver em dry-run, o sistema não usará rede e o custo registrado será 0. | `TestReescrita::test_dry_run_nao_gasta`; o modo real exige `--real --max-chamadas --teto-brl` |
| V6-05 | QUANDO reescrever uma resolução, o sistema preservará enunciado, alternativas e gabarito, e guardará a original no registro de proveniência. | `TestReescrita::test_fluxo_aprova_e_preserva_a_questao_original` |
| V6-06 | SE o professor discordar do gabarito, ENTÃO o item vai para revisão humana sem troca de gabarito. | `TestReescrita::test_professor_que_discorda_*` |
| V6-07 | A resolução nova só será aceita SE: não citar letra; não introduzir número fora do enunciado, das alternativas, da resolução original ou dos resultados; não perder a verificabilidade; chegar ao gabarito; e o avaliador aprovar. | `TestReescrita::test_validar_nova_*`, `test_perder_a_verificabilidade_rejeita`, `test_avaliador_que_reprova_rejeita` |
| V6-08 | Cada item da candidata terá vínculo com a origem: hash da questão v5, ou registro de injeção com modelos, versões de prompt e vereditos. | teste de proveniência da montagem (a escrever junto com `montar_candidata_v6`, portão P3) |
| V6-09 | Nenhuma questão real (`real_banco`, `real_L2`) será removida ou alterada fora das operações autorizadas: a reescrita de resolução e a quarentena com revisão humana registrada. | idem V6-08 |
| V6-10 | A candidata não terá duplicata exata nem contaminação com `val*.jsonl`. | `perfil_base` (duplicatas) + filtro `contamina_val` da injeção + teste da montagem |
| V6-11 | O schema de cada exemplo continuará idêntico ao da v5: `messages[3]`, JSON `{"questoes":[...]}` e campos da questão. | teste da montagem |
| V6-12 | As métricas de qualidade não pioram em silêncio. A candidata só passa se, contra a v5 (`perfil_base`, `triar_resolucoes`, `auditar_distratores`, `auditar_gabaritos_base`), todas estas condições valerem: <ul><li>REFUTADO = 0 e FRACO = 0;</li><li>`matematicamente_problematica` = 0;</li><li>% verificável ≥ 61,9;</li><li>`correta_pouco_explicativa` < 568;</li><li>pares near-dup ≤ 45;</li><li>subtemas vazios < 24 (com o classificador confiável);</li><li>entropia de raciocínio da base ≥ 0,654;</li><li>cada letra entre 18% e 22%;</li><li>explicabilidade dos distratores ≥ 60,3% nos itens novos.</li></ul> | relatório antes/depois (portão P3) |

## Protocolo de comparação de modelos (antes de adotar um modelo num papel)

1. Fixar uma amostra com `melhorar_resolucoes.py --amostra-fixa 20`. A seed é 20261010, então os 20 itens são sempre os mesmos.
2. Rodar a mesma amostra com `sabiazinho-4`, `sabia-4` e `sabia-4-thinking` no papel professor. O avaliador fica fixo em `sabia-4`.
3. Comparar:
   - taxa de aprovação pelos verificadores determinísticos;
   - taxa de aprovação do avaliador;
   - custo real e latência, que vêm do log;
   - leitura humana de 10 pares original × nova.
4. Adotar o modelo mais barato que não perca mais de 1 item aprovado em 20 para o melhor. Registrar a decisão aqui com data.

Nenhum nome de modelo é trocado em silêncio: o papel é fixado por flag (`--modelo-professor`, `--modelo-avaliador`) e gravado em cada registro.

## Protocolo de orçamento

- **Tetos obrigatórios:**
  - toda execução real exige `--max-chamadas` e `--teto-brl` (reescrita) ou `--max-chamadas` (injeção, infraestrutura já existente);
  - o teto em R$ reserva o pior caso de cada envio (entrada estimada + `max_tokens` de saída).
- **Interrupção sem decisão:** teto atingido interrompe o lote. Nada é decidido por falta de resposta, e a próxima rodada retoma pelo cache.
- **Preços:** ficam em `ciclo_v6/precos_maritaca.json`, com fonte e data; nenhum preço está no código.
- **Registro de cada chamada** (`outputs/melhoria_v6/uso_api_*.jsonl`):
  - modelo, papel e versão do prompt;
  - hash do exemplo e `codigo_item`;
  - tokens de entrada e saída;
  - latência e custo;
  - `finish_reason`;
  - uma linha de resultado com o motivo.

## Portões (autorização humana)

| Portão | O que libera | Status |
|---|---|---|
| P0 | auditoria determinística, ferramentas e testes (sem custo) | **feito** |
| P1 | lote pago R: protocolo de modelos (cerca de 120 chamadas, cerca de R$ 3) e depois a reescrita de cerca de 655 resoluções | aguardando autorização |
| P2 | lote pago I: injeção direcionada (40 habilidades, 199 slots) | aguardando autorização e decisão D3 |
| P3 | montagem de `train_curado_v6.jsonl` + manifesto + relatório antes/depois + amostra humana de 40 itens da candidata | depois de P1 e P2 |
| P4 | treino v6 (plano em `RELATORIO_CICLO_V6.md` §15) | depois de P3 aprovado |

## Decisões pendentes

| ID | Pergunta | Recomendação |
|---|---|---|
| D1 | Professor da reescrita: `sabia-4` ou `sabia-4-thinking`? | decidir pelo protocolo acima |
| D2 | Substituir no treino as 445 justificativas oficiais "Correto - conta" do lote L2 pela explicação nova? | sim, para as aprovadas; a original fica na proveniência |
| D3 | Os prompts da injeção passam a levar representação e restrições? Isso muda `VERSAO_PROMPTS` e invalida a reutilização do cache dos juízes. | sim, num lote separado e medido |
| D4 | A alternativa E artificial dos 679 itens reais: manter ou trocar por um 5º distrator plausível? Trocar altera alternativas de item real e exige revisão humana. | manter na v6; piloto de 30 itens com revisão humana para a v7 |
| D5 | Rótulo de dificuldade: dos 473 injetados pedidos Difícil, o revisor julgou só 5 como Difícil; dos 925 pedidos M/D, julgou 57% Fácil. Mas a D3 está suspensa porque ele diz "Fácil" para 83% dos M/D reais do banco. | não reetiquetar; medir na amostra humana |

## Decisões registradas (2026-10-10)

- **D1 (resultado do protocolo R0, 20 itens fixos, avaliador `sabia-4`):** aprovados `sabiazinho-4` 18, `sabia-4` 19, `sabia-4-thinking` 18; 17 itens aprovados pelos três. Custo real: R$ 0,016 / 0,083 / 0,271 no professor. Regra do protocolo → **`sabiazinho-4`** (perde só 1 item para o melhor e custa cerca de 5 vezes menos que o `sabia-4`). Pendente antes do lote R: leitura humana de 10 pares original × nova. Dados em `outputs/melhoria_v6/r0/`.
- **D2:** sim, substituir as justificativas "Correto - conta" do L2 pela explicação nova aprovada; a original fica na proveniência.
- **D3:** sim, a injeção leva representação e restrições, em lote separado e medido (muda `VERSAO_PROMPTS`).
- **D4:** manter a alternativa E artificial na v6; piloto de 30 itens com revisão humana na v7.
- **D5:** não reetiquetar a dificuldade; medir na amostra humana.

## Exceção registrada (2026-10-11)

- **V6-12i** (explicabilidade dos distratores dos itens novos ≥ 60,3%): a v6 final usa os 114 itens do prompt reajustado (`data/injecao_v6b.jsonl`, 55,4%; prompt antigo: 51,9%; injeção da v5: 56,6%). O critério NÃO foi alterado; a falha foi aceita explicitamente pelo responsável do projeto, que escolheu montar a v6 com esses itens. Os demais 14 critérios passam. A v6 de rodadas 1 e 2 (143 itens, prompt antigo) ficou em `data/train_curado_v6_rodadas1e2_prompt_antigo.jsonl`.
- Itens conceituais não entram na v6 (sem 2º revisor de outra família, pois não se usou Gemini nesta rodada).
