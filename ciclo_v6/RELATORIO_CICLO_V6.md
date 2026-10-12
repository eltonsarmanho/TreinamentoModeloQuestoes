# Ciclo v6: auditoria da v5 e plano de melhoria da base

Data: 2026-10-10. Fase concluída: **P0**, ou seja, auditoria determinística, ferramentas e testes. Nenhuma chamada paga foi feita, a base não foi alterada e não houve commit.

A especificação e os critérios de aceite estão em `ciclo_v6/SPEC.md`. Os artefatos regeneráveis ficam em `outputs/melhoria_v6/` e o manifesto em `ciclo_v6/MANIFESTO_auditoria.json`.

**Uso das skills.** Todas as cinco estão instaladas e tiveram as instruções lidas.
- **tlc-spec-driven:** foi aplicada a especificação com critérios EARS, IDs e portões. O commit atômico por tarefa que ela exige **não** foi feito, porque o invariante 15 proíbe commit.
- **dataset-curation:** proveniência, deduplicação, contaminação e piso de dados reais.
- **lora-qlora-recipes:** usada só no plano da §15. A configuração atual já segue a receita.
- **checkpoint-promotion:** usada só no plano da §15. O gate do projeto prevalece.
- **eval-harness:** trata de avaliações de sessões do Claude Code e as ferramentas dela (`scripts/eval-harness.js`) não existem neste repositório. Apliquei só o princípio de definir os critérios antes de implementar e manter testes de regressão.

---

## 1. Diagnóstico da v5, a partir dos arquivos reais

Os números informados pelo responsável **conferem** com o arquivo:

| Indicador | Informado | Medido |
|---|---|---|
| exemplos / questões | 2.698 / 2.740 | 2.698 / 2.740 |
| letras do gabarito | 20% cada | A 548, B 549, C 547, D 548, E 548 |
| CONFIRMADO / JUIZES / CEGO / FRACO / REFUTADO | 1716 / 922 / 102 / 0 / 0 | idem (`AUDITORIA_GABARITOS_v5`) |
| resoluções < 40 caracteres | 519 | 519 (445 `real_L2`, 73 sintéticos, 1 do banco) |
| E = "Nenhuma das alternativas anteriores" | 681 | 681 literais; 692 pela classe NDA de `schema_utils` (11 injetados com variações) |
| habilidades com < 30 questões | 22 | 22 |
| contaminação exata com `val`, `val_frozen` e `val_novos` | — | 0 / 0 / 0 |

### 1.1 O que já existe e funciona
- **Pipeline de agentes**:
  - cliente medido, orçamento duro de chamadas, cache dos juízes e sanitização de segredos;
  - validador e revisor cegos com permutação, e 2º revisor Gemini opcional;
  - filtros de estrutura, visual, near-dup, contaminação, D5 e geometria.
- **Classificador de diversidade** (`diversidade.py`): subtema, raciocínio, estrutura, representação e contexto, com planejador de slots.
- **Relatório de cobertura e déficit** (`relatorio_base.py`), que já roda sobre a v5 sem mudança.
- **Auditoria de gabarito** em classes (`auditar_gabaritos_base.py`) e balanceamento de letras.
- **Gate multiseed** e amostra humana cega (`amostra_humana.py`).

### 1.2 O que está incompleto ou fora de sincronia
- **Juiz truncado** (corrigido neste ciclo): resposta cortada por `max_tokens` era aceita se o JSON final fechasse. Isso valia para o validador, a fase 2 e o revisor, e aconteceu em 10 de cerca de 9 mil respostas nos logs.
- **Gemini fora do ar:** a queda do 2º revisor deixa o item entrar só com validador + revisor, marcado `revisor2: None` (`agentes_questoes.py:2172`). É uma degradação documentada ("degrades gracefully"), e não a alterei. Nas 7 habilidades conceituais, porém, o plano v6 exige o 2º revisor como nível de evidência. A montagem da candidata (P3) deve recusar item conceitual com `revisor2: None`, em vez de tratar a queda do Gemini como aprovação.
- **Log de uso** (`uso_api.jsonl`): não registra versão do prompt, hash do item, custo nem resultado. Os papéis novos registram, via `sabia_v6.py`.
- **Custo:** não há orçamento em R$, só teto de chamadas e tokens. Os papéis novos têm teto em R$.
- **Slots da injeção:** não levam representação nem restrições, e o campo `estrutura` é planejado mas não chega ao gerador.
- **Geometria no relatório:** o gancho de `relatorio_base.py` (`_veredito_geometria`) procura funções que não existem, então todo item fica "sem_veredito". Não alterei, para não mudar o déficit sem aviso; o veredito de geometria está nesta auditoria pelo `perfil_base`.
- **Divergências no `CLAUDE.md`:**
  - os CLIs pagos não têm `--real` (o real é o padrão, e a trava é `--max-chamadas`);
  - a fase 2 do validador vê o gabarito e a resolução do autor;
  - a injeção grava `data/injecao_saeb.jsonl` e não a v3;
  - o verificador de geometria **reprova** na injeção (só é sombra na inferência).

  Corrigido no `CLAUDE.md`.
- **3 testes já falhavam antes deste ciclo:**
  - `test_modelo_padrao_do_revisor_e_thinking` exige `run_base_conhecimento.sh`, que está fora do checkout porque o `.gitignore` exclui `*.sh`;
  - outros 2 verificam uma pré-condição de corpus desatualizada: o viés medido caiu de 38,5% para 31,2%, e o teste exige mais de 35%.

  Não mexi nesses três.

### 1.3 Reutilizado, não duplicado
- **Para medir:** `relatorio_base.py` (cobertura e déficit), `diversidade.py` (classificador, near-dup e planejador), `injetar_questoes.lacunas` / `planejar_slots` e `schema_utils` (verificador, NDA e valores de alternativa).
- **Para a parte paga:** `auditar_gabaritos_base` (classes e origem) e o cliente, `Orcamento`, `_sanitizar`, `_retentavel` e `abrir_para_gravar` de `agentes_questoes`. O prompt do avaliador é o de **fase 2 do validador** (`VALIDADOR_FASE2`), já calibrado.
- **Para registrar e verificar:** `congelar_baseline` (hash e git) e `padronizar_enunciado` (caixa alta).

---

## 2. Lacunas encontradas e evidências

| # | Dimensão | Evidência | Afetados |
|---|---|---|---|
| A | Cobertura curricular | 24 subtemas da taxonomia sem nenhum item (`perfil_v5`); 40 de 79 habilidades abaixo da meta, com déficit total de 213 (`relatorio_base_v5`); 5º H01 e 9º H01 cobrem só 1 de 5 e 1 de 4 subtemas | 40 habilidades |
| B | Subtemas ausentes | 9 habilidades com 50% ou mais em "outros" (2º H19 e H21, 3º EF03MA01, 5º H08 e H21, 9º H05, H10, H13 e H25): o classificador léxico falha e não há prova de lacuna | 9 habilidades |
| C | Dificuldade | rótulos F/M/D 1078/855/807; dos 473 injetados pedidos Difícil, o revisor julgou só 5 como Difícil. **Mas** a D3 está suspensa porque o revisor diz "Fácil" para 83% dos M/D reais, então a evidência é fraca | hipótese |
| D | Raciocínio repetitivo | 12 habilidades com entropia de raciocínio < 0,3; na base, 0,654; 871 questões (31,8%) com tipo "indefinido" pelo classificador | 12 habilidades |
| E | Resoluções curtas | 568 "corretas, porém pouco explicativas" (as 519 com menos de 40 caracteres estão todas aqui) e 87 incompletas | 655 |
| F | Distratores | explicabilidade por erro comum de 60,3%: 82,6% nos `real_L2` e 52,8% a 56,6% nos destilados e injetados. 15 habilidades abaixo de 20% (5º H17 e 9º H14 com 0%, porque as alternativas são textuais) | itens de valor |
| G | NDA artificial | presente em 100% dos itens reais (679, sempre a E, nunca o gabarito): é um distrator "gratuito" que reduz a questão a 4 opções reais | 679 |
| H | Duplicatas | 0 exatas; 45 pares near-dup: 30 de variação só numérica, 10 duplicatas prováveis e 5 a revisar. Só 2 pares envolvem item real | 45 pares |
| I | Evidência limitada | 1.043 questões (38,1%) não verificáveis pela aritmética; 922 sustentadas só por dois juízes da mesma família (Sabiá) | 922 |
| J | Ambiguidade de alternativa | DIST-H01-Fácil-00406: "4,05" e "4.05" como alternativas distintas | 1 |

---

## 3. Plano priorizado de melhoria

A prioridade de cada habilidade não vem de uma meta uniforme. A fórmula (`planejar_injecao_v6.prioridade`) é:

```
prioridade = 3·(subtemas vazios) + 2·(fração da meta faltando) + 1·(concentração > 60%) + 1·(1 − entropia de raciocínio)
```

Subtemas vazios e concentração são descontados pela fração que o classificador deixou em "outros".

| Prioridade | Lacuna | Intervenção | Risco | Validação | Custo (API) |
|---|---|---|---|---|---|
| 1 | E: resoluções | Reescrever **só a explicação** dos 655 itens da fila; enunciado, alternativas e gabarito ficam congelados | baixo: o original fica, e qualquer falha mantém o original | verificadores determinísticos + avaliador fase 2 + amostra humana | cerca de 1.500 chamadas, R$ 10–14 (sabia-4) ou R$ 33–50 (thinking); protocolo de modelos antes, cerca de R$ 3 |
| 2 | A, D: cobertura e raciocínio | Injeção direcionada: 40 habilidades e 199 slots, com 5º H01, 9º H01 e 2º H11 no topo | médio: itens novos | pipeline de agentes inteiro + nível de evidência por slot | cerca de 1.130 chamadas, cerca de R$ 42 (teto prudente R$ 64) + Gemini nas 7 habilidades conceituais |
| 3 | H: duplicatas | Remover só os near-dups **não reais**, pela regra existente (`curar_diversidade`) | baixo | `perfil_base` antes/depois | 0 |
| 4 | J e revisão | 1 item de distrator para revisão humana | baixo | professor | 0 |
| 5 | B: classificador | Revisar palavras-chave das 9 habilidades com "outros" ≥ 50%. Isso muda a `versao_regua` e exige registro no G11 | médio | perfil antes/depois | 0 |
| 6 | G: NDA artificial | v6: manter; v7: piloto de 30 itens com 5º distrator gerado e revisado por humano | alto: altera item real | revisão humana | baixo |
| 7 | C: dificuldade | Não reetiquetar; medir na amostra humana | — | professor | 0 |
| 8 | F: distratores dos novos | Restrição nos slots: erros conceituais da habilidade, sem NDA de preenchimento. Exige mudar o prompt do gerador (decisão D3) | médio | `auditar_distratores` nos novos ≥ 60,3% | incluso em 2 |

Intervenções de alto risco (6) e mudanças de prompt (8) têm portão próprio (`SPEC.md`).

---

## 4. Arquivos existentes reutilizados

`relatorio_base.py`, `diversidade.py`, `injetar_questoes.py` (`lacunas`, `planejar_slots`, CLI de injeção), `agentes_questoes.py` (cliente, orçamento, cache, sanitização, prompt de fase 2), `auditar_gabaritos_base.py`, `schema_utils.py`, `padronizar_enunciado.py`, `congelar_baseline.py`, `curar_diversidade.py` (regra de near-dup), `amostra_humana.py`, `promover_checkpoint.py`.

## 5. Arquivos novos ou modificados

| Arquivo | Tipo | Justificativa |
|---|---|---|
| `src/perfil_base.py` | novo | Mede as dimensões que o `relatorio_base` não cobre: resolução, NDA, letra, dificuldade rótulo × julgada, near-dup auditável com trigramas, entropia por dimensão normalizada pelo universo da taxonomia, visual, verificabilidade, classe de auditoria |
| `src/auditar_distratores.py` | novo | Não havia nenhuma métrica de distrator; leitura de valor própria, porque `schema_utils` lê "5.000" como 5 |
| `src/triar_resolucoes.py` | novo | Triagem nas 5 classes pedidas; o comprimento é só um sinal |
| `src/sabia_v6.py` | novo | Camada fina sobre o cliente e o orçamento existentes: teto em R$ com reserva do pior caso, log completo, cache por versão de prompt, falha sempre fechada. `Agentes` só aceita os 4 papéis da injeção |
| `src/melhorar_resolucoes.py` | novo | Reescrita da explicação (dry-run por padrão; `--real` exige `--max-chamadas` e `--teto-brl`) |
| `src/planejar_injecao_v6.py` | novo | Plano priorizado com representação, restrições e nível de evidência; reusa `lacunas` e `planejar_slots` |
| `src/manifesto_v6.py` | novo | Proveniência: hashes, versões de verificadores, prompts e régua, relatórios, candidata |
| `src/agentes_questoes.py` | modificado | Validador (fases 1 e 2) e revisor reprovam com `resposta_truncada` quando `finish_reason="length"` (invariante 8). `VERSAO_PROMPTS` e `VERSAO_JUIZES` não mudam: o cache nunca guardou resposta truncada |
| `tests/test_base3_decisoes.py` | modificado | 2 asserções de `TestP4CacheSemTruncada` descreviam o comportamento antigo (fase 1 truncada seguia para a fase 2). A intenção do teste foi mantida (truncada não vai ao cache e é refeita) e a asserção agora exige reprovação |
| `tests/test_entradas_imutaveis.py`, `tests/test_juiz_truncado.py`, `tests/test_ciclo_v6.py` | novos | 37 testes, em §12 |
| `ciclo_v6/ENTRADAS_IMUTAVEIS.json`, `precos_maritaca.json`, `SPEC.md`, este relatório, `MANIFESTO_auditoria.json` | novos, versionados | trava, preços, especificação, evidência |
| `CLAUDE.md` | modificado | Seção do ciclo v6 e correção das 4 divergências de §1.2 |

---

## 6. Agentes e prompts da API Sabiá

| Papel | Modelo padrão (flag) | Entrada | Saída (JSON) | Decide |
|---|---|---|---|---|
| `professor_resolucao` | `sabia-4` (`--modelo-professor`), temp. 0,2, 900 tokens; thinking: 4.096 | ano, habilidade, enunciado, alternativas e **gabarito** (já certificado) | `{"discorda": bool, "motivo", "resolucao"}` | só a explicação; se discordar, vai para revisão humana |
| `avaliador_resolucao` | `sabia-4` (`--modelo-avaliador`), temp. 0, 300 tokens | prompt **existente** da fase 2 do validador + resolução nova | `resolucao_correta`, `usa_dado_ausente`, `justificativa_falsa`, `chega_na_alternativa` | aprova ou rejeita a resolução nova |
| injeção (existente) | gerador, validador e revisor `sabia-4-thinking`; fase 2 `sabia-4`; 2º revisor `gemini-2.5-flash` | — | — | inalterado |

O prompt do professor está em `melhorar_resolucoes.PROFESSOR_SISTEMA` e tem versão em `sv.versao_prompt` (sha1 de 12 caracteres) gravada em cada registro.
- **Regras do prompt:** usar só dados do enunciado; não mudar número; contas explícitas; não citar letras; no máximo 6 frases.

**Independência.** O professor e o avaliador são da mesma família (Sabiá), então a aprovação **não** é evidência independente (invariante 7). Por isso:
- o gabarito não está em jogo;
- as checagens determinísticas valem como trava: número inventado, verificabilidade, citação de letra, chegar ao gabarito;
- a amostra humana de P3 inclui resoluções reescritas.

**Modelo.** Sabiazinho-4 só entra se vencer o protocolo de comparação (`SPEC.md`). Nenhum modelo é trocado em silêncio.

## 7. Protocolo de orçamento e estimativa de custo

Os preços estão em `ciclo_v6/precos_maritaca.json`, em R$ por milhão de tokens de entrada/saída:

| Modelo | Entrada | Saída |
|---|---|---|
| sabia-4 | 5 | 20 |
| sabia-4-thinking | 5 | 40 |
| sabiazinho-4 | 1 | 4 |

Os tokens por papel vêm das medianas dos logs reais: validador thinking 3.180/630, revisor 2.625/850, fase 2 470/54, gerador 1.950/920.

| Lote | Escopo | Chamadas | Custo previsto | Teto proposto |
|---|---|---|---|---|
| R0, protocolo de modelos | 20 itens × 3 modelos × (professor + avaliador) | cerca de 120 | cerca de R$ 2–3 | `--max-chamadas 130 --teto-brl 5` (por modelo: 45 / R$ 2) |
| R, reescrita | 655 itens | cerca de 1.506 (com 15% de retentativa) | R$ 10,30 mediano / R$ 14,08 p90 (sabia-4); R$ 32,74 / R$ 50,23 (thinking) | `--max-chamadas 1700 --teto-brl 20` (sabia-4) |
| I, injeção | 40 habilidades, 199 slots | cerca de 1.128 (5,67 por aceita, medido) | R$ 42,45 (medido R$ 0,213 por aceita: R$ 286,65 / 1.344) | `--max-chamadas 1700`; 2º revisor Gemini `--max-chamadas-gemini 60` |

**Regras:**
- teto duro antes de cada envio;
- teto atingido não decide nada;
- cache e retomada;
- falha de API vira abstenção;
- o dry-run roda sem rede.

O dry-run da reescrita rodou com sucesso: 655 itens, custo 0.

## 8. Cobertura antes/depois

**Antes (v5):**
- 79 habilidades, 228 de 252 subtemas cobertos, déficit de 213 em 40 habilidades;
- 22 habilidades com menos de 30 questões;
- 18 habilidades com K=1, em que só o raciocínio e o contexto podem variar.

O **depois** só existe depois de P1 a P3. O critério de aceite é o V6-12 do `SPEC.md`, e a tabela completa por habilidade está em `outputs/melhoria_v6/perfil_v5.md` e `relatorio_base_v5.md`.

## 9. Qualidade das resoluções

| Classe | Questões |
|---|---|
| Adequada | 2.085 |
| Correta, porém pouco explicativa | 568 |
| Incompleta | 87 |
| Matematicamente problemática | 0 |
| Não avaliável | 0 |

- **Por origem:**
  - `real_L2`: 445 pouco explicativas, o estilo "Correto - conta";
  - sintéticos: 98 pouco explicativas;
  - injetados: 69 incompletas e 15 pouco explicativas;
  - destilados: 14 incompletas e 5 pouco explicativas.
- **Citação de letra:** 346 resoluções citam letra; em 342 ela é o próprio gabarito, e as 4 restantes são comentários sobre distratores. A permutação da v5 manteve as referências corretas.
- **Ponto cego:** o verificador aritmético não detecta conta errada que ele não consegue ler ("35 + 18 = 43" dá "sem_conta"). "0 problemáticas" não prova que não haja conta errada nas resoluções.

## 10. Qualidade dos distratores

| Origem | Explicabilidade | Espaço de valor | NDA | Correta mais longa |
|---|---|---|---|---|
| destilado | 52,8% | 73,2% | 0,3% | 0,6% |
| injetado | 56,6% | 50,1% | 0,8% | 1,0% |
| real_L2 | 82,6% | 98,4% | 100% | 0% |
| real_banco | 51,2% | 65,4% | 100% | 1,7% |
| sintético | 55,7% | 100% | 0% | 0% |

- **Indicadores da base:**
  - valor repetido com a mesma unidade: 0,6%;
  - espaço heterogêneo: 1,3%;
  - unidade inconsistente: 0,5%.
- **Revisão humana:** 1 item, `DIST-H01-Fácil-00406` ("4,05" × "4.05").
- **Falsos positivos eliminados durante a auditoria:**
  - "8 ovos" × "8 galinhas", que é o mesmo número com objeto diferente;
  - "5" × "5.000", que é milhar pt-BR;
  - "≤" × "≥", que a normalização ASCII apagava;
  - frações equivalentes quando o enunciado pede a forma irredutível.

## 11. Diversidade semântica

A entropia está normalizada pelo universo de categorias que a taxonomia admite:

| Dimensão | Entropia |
|---|---|
| subtema (252 combinações ano × habilidade × subtema) | 0,943 |
| raciocínio | 0,654 |
| estrutura | 0,589 |
| representação | 0,492 |
| contexto | 0,929 |

- **Contexto não é diversidade matemática:** o contexto já está quase uniforme (0,929), enquanto raciocínio e representação estão concentrados. Por isso o plano gira subtema × raciocínio e não só o contexto.
- **Valores numéricos:** 63,6% dos itens têm um conjunto de números distinto.
- **Variação só numérica:** 19 enunciados repetem outro a menos dos números (base inteira).
- **Por habilidade:** a tabela fica em `perfil_v5.md`, e os 45 pares near-dup com os trigramas coincidentes ficam em `near_dups_v5.jsonl`.
- **Repetições no plano de injeção:** 50 dos 199 slots repetem um par (subtema, tipo). Ocorre só em habilidades com poucas combinações; 9º H17, por exemplo, tem 3 combinações para 18 slots.

## 12. Testes executados e resultados reais

| Momento | Resultado |
|---|---|
| Antes de qualquer alteração | 758 passaram, 3 falharam (as 3 já existentes de §1.2) |
| Depois | **795 passaram, 3 falharam** (as mesmas 3) |

Testes novos (37):
- `test_entradas_imutaveis.py` (2): v5, v3, train, val, val_frozen e DB conferem com o sha travado;
- `test_juiz_truncado.py` (5): validador fases 1/2 e revisor reprovam saída truncada, com controle sem truncagem;
- `test_ciclo_v6.py` (30): entropia pelo universo; metadado ausente vira "desconhecido"; distratores (unidade, milhar, notação, forma irredutível, inequações, explicabilidade); triagem nas 5 classes; teto de chamadas e de R$ bloqueia antes de enviar; real exige teto; falha, truncada ou ininterpretável vira abstenção; cache não paga duas vezes; log tem os campos exigidos; truncada não entra no cache; reescrita preserva a questão e a proveniência; discordância vai para humano; falha nunca aprova; avaliador que reprova rejeita; dry-run custa 0; prioridade desconta classificador falho; nível de evidência.

O `baseline_v3/MANIFESTO.json` também confere com o disco.

## 13. Manifesto de proveniência e hashes

O manifesto está em `ciclo_v6/MANIFESTO_auditoria.json`. Ele registra:
- commit, arquivos alterados e as 6 entradas imutáveis (todas intactas);
- `VERSAO_PROMPTS` 3358bf312d22, `VERSAO_JUIZES` j2-d5-validador-g e `VERSAO_FILTROS` f6-h2-d5-extenso-planilha;
- régua de diversidade 4dd53baebbad…;
- sha256 do código dos verificadores, da gramática, da taxonomia e dos preços;
- sha256 dos 13 relatórios.

O manifesto da candidata (`--fase candidata`) acrescenta o sha da v6 e as contagens por operação.

## 14. Critérios de aprovação da candidata

São os critérios V6-01 a V6-12 do `SPEC.md`. Em resumo:
- **Gabarito e itens reais:** nenhum gabarito trocado; nenhuma questão real removida ou alterada fora das duas operações autorizadas.
- **Qualidade mínima:** 0 REFUTADO, 0 FRACO, 0 problemáticas; % verificável ≥ 61,9.
- **Melhoria:** menos "pouco explicativas" que 568; menos subtemas vazios que 24 (classificador confiável); near-dup ≤ 45; entropia de raciocínio ≥ 0,654.
- **Integridade:** letras entre 18% e 22%; 0 duplicata exata e 0 contaminação; schema idêntico; proveniência de todos os itens.
- **Revisão humana:** amostra cega de **40 itens da candidata** (20 resoluções reescritas e 20 injetados) corrigida por professor, com validade não pior que a da v5 (intervalo de Wilson).

## 15. Plano de treinamento (posterior, só depois de P3 aprovado)

1. **Isolar o efeito dos dados.** A v5 ainda não foi treinada. São três braços, com os **mesmos** hiperparâmetros do `baseline_v3/MANIFESTO.json`:
   - r=16, α=32 (=2r), LR 2e-4, batch 2 × acumulação 8 (efetivo 16);
   - 3 épocas, seed 42, `train_on_responses_only`, `MAX_SEQ_LENGTH` 1024;
   - mesma exportação Q4_K_M.

   | Braço | Treino |
   |---|---|
   | baseline | v3 (já existe) |
   | A | v5 |
   | B | v6 |

   Comparações: v3 × A mede a consolidação; A × B mede o ciclo v6. Nenhum hiperparâmetro muda junto. A configuração atual já segue a receita da skill lora-qlora (all-linear, α=2r, LR 2e-4, batch efetivo < 32), então não há ajuste de receita a misturar.
2. **Avaliação:**
   - mesma `val_frozen_v1`, mesmos prompts e seeds pareadas (`--seed-rodada 0,1,2`);
   - modo do app: planejado e com sufixo;
   - mesmo motor (`llama-server`, `-c 2048`, `-np 1`).
3. **Gate.** `promover_checkpoint.py --perfil multiseed` **sem alterar critérios**. O gate cobre:
   - acerto e consistência (G2/G3 por McNemar) e viés de letra (G6 agregado);
   - aderência e latência;
   - observações não bloqueantes: cobertura, variabilidade entre rodadas, NDA e caixa alta.
4. **Além do gate:**
   - `auditar_distratores` e `triar_resolucoes` sobre as questões geradas;
   - near-dups, cobertura semântica e robustez entre seeds;
   - amostra humana cega de 40 questões geradas (`amostra_humana.py`, procedimento existente).
5. **Promoção.** Só com PROMOVIDO no gate. A skill checkpoint-promotion sugere um conjunto de deriva de capacidade geral (MMLU/GSM8K), que o projeto não usa. Fica como observação, não como critério novo (invariante 14).

---

## Próximo passo: autorização necessária

Nada pago foi executado. Para seguir:
- **P1-R0:** protocolo de modelos, até 130 chamadas e R$ 5;
- **P1-R:** reescrita das resoluções, até 1.700 chamadas e R$ 20 com sabia-4, ou R$ 55 com thinking;
- **P2:** injeção, até 1.700 chamadas Maritaca e 60 Gemini, cerca de R$ 42 a R$ 64. Antes, a decisão D3: o prompt passa a levar representação e restrições?
- **Decisões** D1, D2, D4 e D5 (`SPEC.md`).
