# SPEC: Diversidade e cobertura semântica (núcleo)

Escopo: `src/diversidade.py`, `src/build_taxonomia.py`, `data/taxonomia_subtemas.json` e `tests/test_diversidade.py`. A integração com `distill_teacher`, `test_model` e `promover_checkpoint` fica para uma etapa posterior. O schema `{"questoes":[...]}`, a gramática e as validações de `schema_utils` não mudam.

## Requisitos (EARS)

| ID | Requisito | Verificação |
|----|-----------|-------------|
| R1 | O sistema SHALL gerar a taxonomia para TODA (ano, habilidade) de Matemática do `DB/questoes.db`, com a chave `"ano\|habilidade"`. | `test_todas_as_habilidades_de_matematica` |
| R2 | WHEN deriva subtemas, o sistema SHALL usar só a `descricao_item` (enumeração entre parênteses, léxico e hiperônimos), sem regras por código H. | `test_mesmo_codigo_significa_coisas_diferentes_por_ano` |
| R3 | O sistema SHALL NOT incluir subtemas incompatíveis com a descrição. Trechos negados ("sem utilizar frações") e restrições ("malha quadriculada" exclui círculo) são respeitados. | `test_nao_inventa_subtema_incompativel` |
| R4 | Os contextos narrativos SHALL ser uma lista global separada dos subtemas. | `test_contextos_sao_globais_e_nao_subtema` |
| R5 | `classificar_questao` SHALL devolver subtema, tipo_raciocinio, objeto_matematico, contexto, estrutura e representacao. Quando nada casa, devolve `outros`, `indefinido` ou `sem_contexto`. | `TestClassificador` |
| R6 | `planejar_lote(N)` SHALL cobrir min(N,K) subtemas distintos, com no máximo ceil(N/K) por subtema (N=2→2, N=3→≥2, N=5→≥3 quando K permite). | `test_tamanhos_de_lote_varios_k` |
| R7 | WHEN K=1, o planejador SHALL variar tipo de raciocínio, contexto e estrutura. | `test_um_subtema_diversifica_outros_eixos` |
| R8 | Com a mesma seed e as mesmas entradas, o plano SHALL ser idêntico. | `test_determinismo_por_seed` |
| R9 | WHEN duas questões diferem só nos números, `duplicate_rate` SHALL contá-las como duplicata. | `test_troca_de_numeros_e_duplicata` |
| R10 | `diversity_score` SHALL expor todos os componentes e os pesos (`PESOS_DIVERSIDADE`). | `test_componentes_do_diversity_score` |
| R11 | WHEN um candidato viola o plano (subtema saturado ou fora do plano, near-dup, contexto repetido), o sistema SHALL produzir uma restrição em português e regenerar até `max_tentativas` vezes. | `TestRegeneracao` |
| R12 | `sufixo_prompt` SHALL ter o formato único ` Subtema: X. Tipo de raciocínio: Y. Contexto: Z.` | `test_sufixo_prompt_formato_unico` |
| R13 | O sistema SHALL NOT usar embeddings nem dependências pesadas (só stdlib) e SHALL NOT aumentar a temperatura. | inspeção de imports |

## Critérios de aceite
- `venv/bin/python -m unittest tests/test_diversidade.py` passa por completo.
- O plano para 9º H17 com N=10 fica em 5 triângulos e 5 quadriláteros, e cada subtema alterna a classificação por lados e por ângulos.
- `build_taxonomia.py` roda sem erro. Conferido à mão: os subtemas de H16 a H22 do 2º, 5º e 9º anos são compatíveis com a descrição de cada habilidade.
