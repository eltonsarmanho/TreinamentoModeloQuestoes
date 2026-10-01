# Plano de melhoria — todos os gates aprovados no modo planejado C

Data: 2026-09-30. Modo de produção decidido: **C** = N chamadas de 1 questão, cada
uma com o sufixo do plano (Subtema / Tipo de raciocínio / Contexto), sem
regeneração por diversidade. Prioridade: qualidade > tempo de retorno no
celular offline.

## 1. Onde estamos (evidência dos nossos próprios relatórios)

Veredito `outputs/relatorios/VEREDITO_planejado.json` (48 pares prompt×seed por
modelo, modo C, modelo antigo `0e5ceb6e` vs novo `082ea677`):

| Gate | Antigo → Novo | Situação |
|---|---|---|
| P1 JSON válido | 100 → 100 | ok |
| P2 Entrega N questões | 100 → 100 | ok |
| P4 Aderência à habilidade | 78,6 → 87,6 % | ok, mas abaixo da meta (≥ 92 %) |
| P5 Depende de visual ausente | 4,4 → 0 % | ok |
| P6 Dificuldade correta | 100 → 100 | ok |
| P7 Sem esquecimento 5º/9º | preservado | ok |
| P8 Tempo por questão (informativo) | 14,2 → 15,8 s (+11 %) | ok no PC; **não medido no celular** |
| **G11 / P3 Gabarito inconsistente** | 1,32 → 2,19 % | **reprovado** (G11 tolera 0 pp) |

Conferência manual das 8 questões marcadas: **3 erros reais no novo, 1 no
antigo, 4 falsos positivos do verificador** (perguntas "qual é maior?" com
gabarito textual). Em 2 dos 3 erros reais do novo a resposta certa não está
entre as alternativas.

Modo N=1 sem sufixo (gate histórico, `VEREDITO_diversidade_vs_atual.json`):
G1, G2, G3, G6 (letra mais frequente 36,7 → 50 %), G7 reprovados. Causa provável:
85 % do `train_curado.jsonl` (1.218 de 1.435) tem o sufixo → o modelo ficou
dependente do condicionamento e regrediu no prompt "cru". O app não usa esse
prompt, mas o viés de gabarito (G6) aparece em qualquer modo.

Diagnóstico em uma frase: **o modelo novo é mais diverso e mais aderente, mas
ainda erra contas/gabarito em ~1,3 % das questões e concentra a resposta numa
letra; a medição tem ruído (31 questões verificáveis em 228) e falsos positivos.**

## 2. Princípios (e a evidência por trás de cada um)

1. **Verificador determinístico + rejeição, antes de qualquer retreino.**
   Filtrar amostras por um verificador e treinar só nas aprovadas é o que
   funciona em raciocínio matemático com modelos pequenos: STaR (Zelikman et
   al., 2022), Rejection-sampling Fine-Tuning (Yuan et al., 2023, "Scaling
   Relationship on Learning Mathematical Reasoning"), e é o laço de
   pós-treino do Llama 2/3 (Meta, 2023/2024). Já fazemos isso no
   `distill_teacher.py`; o plano aperta o verificador.
2. **Verificador na inferência também (best-of-N com checagem).** Cobbe et al.
   (2021, GSM8K) mostram que um verificador reordenando amostras rende mais que
   escalar o modelo. O `generate_validated` já é best-of-N; falta a checagem
   "resposta ∈ alternativas".
3. **Qualidade > quantidade nos dados de SFT.** LIMA (Zhou et al., 2023) e a
   família Phi ("Textbooks Are All You Need", Gunasekar et al., 2023): poucos
   exemplos limpos e variados batem muitos ruidosos. Destilação de professor
   forte com explicações passo a passo (Orca, Mukherjee et al., 2023) é o que
   o Sabiá-4-thinking nos dá.
4. **Não depender do condicionamento: dropout do sufixo.** Análogo ao
   classifier-free guidance (Ho & Salimans, 2022): treinar com o condicionamento
   removido numa fração dos exemplos mantém o modelo bom com e sem ele. Resolve
   a regressão N=1 e reduz fragilidade se o app mudar o prompt.
5. **Viés de posição da resposta é artefato de dados.** Modelos de múltipla
   escolha herdam a distribuição da letra correta do treino (Zheng et al.,
   2023, "Large Language Models Are Not Robust Multiple Choice Selectors").
   Correção barata: permutar alternativas no treino até a letra ser uniforme.
6. **Preferências só se o SFT não bastar.** DPO (Rafailov et al., 2023) / ORPO
   (Hong et al., 2024) com pares "questão consistente vs inconsistente" que o
   nosso filtro já produz de graça. Implementação: TRL (github.com/huggingface/trl),
   compatível com Unsloth.
7. **Decisão pré-registrada e pareada.** Mesmas seeds, McNemar/Wilson, tolerâncias
   fixadas ANTES de rodar. Já é a regra do `promover_checkpoint.py`; o plano só
   corrige o conflito P3 vs G11 para o próximo ciclo.

Ferramentas de referência: unsloth (treino), llama.cpp (GBNF, `--grammar-file`),
llama.rn (React Native), TRL (DPO/ORPO), Outlines / lm-format-enforcer
(decodificação restrita, caso a GBNF precise ser trocada — hoje não precisa).

## 3. Fases

### Fase 0 — Corrigir o instrumento (sem retreino, ½ dia)

Sem isso, qualquer melhoria fica invisível dentro do ruído.

| # | Ação | Arquivo | Efeito esperado |
|---|---|---|---|
| 0.1 | `check_consistency`: perguntas de comparação ("qual é maior/menor", "quem tem mais") com gabarito textual passam a ser verificadas pela ordem dos resultados calculados, não pelo valor literal | `schema_utils.py` | remove os 4 falsos positivos vistos |
| 0.2 | Nova checagem `resposta_nas_alternativas`: se a resolução termina num número e ele não está em nenhuma alternativa → inconsistente | `schema_utils.py` | pega 2 dos 3 erros reais do novo |
| 0.3 | Unificar tolerância P3 = G11 para gabarito inconsistente (**vale só a partir deste ciclo**, registrado como revisão pós-veredito, como foi feito com G4) | `promover_checkpoint.py` | fim do conflito entre gates |
| 0.4 | Aumentar amostra verificável: 16 → 32 prompts (mais prompts com contas explícitas: H06, H08, H12, H20, H24), 3 seeds | `data/prompts_diversidade.json` | n verificável 31 → ~120; resolução de ~1 pp |
| 0.5 | Reportar intervalo de Wilson e McNemar nos gates P3/P4 | `promover_checkpoint.py` | reprova só piora significativa |

Critério de saída: 0 falsos positivos numa amostra manual de 40 questões marcadas.

### Fase 1 — Guardas de inferência (sem retreino, 1 dia)

Entram no `generate_validated` e, portanto, no app (mesma lógica em TypeScript).

| # | Ação | Custo de tempo | Efeito |
|---|---|---|---|
| 1.1 | Rejeitar candidato com resposta fora das alternativas (0.2) e tentar de novo (já há `retries`) | +1 chamada só quando falha (~1–2 % das questões) | inconsistentes reais 1,3 % → ~0,4 % |
| 1.2 | Rejeitar alternativas duplicadas/"Nenhuma das anteriores" repetida (caso base_k0_s1 P08) | zero | G1 alternativas_distintas |
| 1.3 | Sortear a posição da resposta correta **depois** de gerar (permutação determinística por seed, reescrevendo a letra) | zero | G6 letra mais frequente → ~20 % em qualquer modelo |
| 1.4 | Cache do prefixo do prompt (system prompt + "Gere 1 questão… Ano… Habilidade…" são iguais nas N chamadas): llama.rn mantém o contexto carregado; usar `n_keep`/prompt cache para reaproveitar o KV do prefixo | **−20 a −35 % de pré-fill por chamada no celular** | tempo |
| 1.5 | Limitar `n_predict` pelo tamanho típico (p95 das respostas aceitas ≈ 350 tokens) em vez de 512 | −10 % no pior caso | tempo |

Reavaliar (Fase 0 + 1) nos dois modelos: se o novo passar em tudo, **promover
aqui** e as fases 2–4 viram o ciclo seguinte.

### Fase 2 — Dados: rodada 3 de destilação, dirigida pelos erros (2–3 dias, usa Maritaca)

Objetivos por evidência: (a) consistência; (b) aderência 87,6 → ≥ 92 %; (c)
independência do sufixo; (d) letra uniforme.

| # | Ação | Detalhe |
|---|---|---|
| 2.1 | Plano de destilação dirigido: `curar_diversidade.py` sobre `train_curado` + erros das avaliações (subtemas onde aderência < 80 % e onde houve inconsistência: 5º H16 comparação de áreas, 5º H21 tabelas, 9º H12 equações, 9º H08 frações) | `data/plano_destilacao_v3.json`, alvo ≈ 600 aceitas |
| 2.2 | Filtro do professor mais duro: 0.2 + "todas as alternativas numéricas distintas e plausíveis (distratores = erros comuns)" + resolução ≤ 900 chars | `distill_teacher.py filtrar()` |
| 2.3 | **Dropout do sufixo**: ao montar o treino, 30 % dos exemplos destilados recebem o prompt SEM sufixo (mesma resposta) | `curar_diversidade.py --sufixo-dropout 0.3` |
| 2.4 | **Permutação de alternativas** no treino: reembaralhar A–E por seed até a distribuição da letra correta ficar uniforme (±3 pp) em cada (ano, habilidade) | `curar_diversidade.py --balancear-gabarito` |
| 2.5 | Guardar os **rejeitados por `resposta_inconsistente`/`dados_ausentes`** pareados com a versão aceita do mesmo slot → `data/preferencias.jsonl` (para a Fase 4, custo zero agora) | `distill_teacher.py --salvar-rejeitados` |
| 2.6 | Exemplos N>1 (`montar_lotes.py`, já pronto): **não** entram — o app usa N=1; evitam-se sequências longas e o teto de 1024 tokens | — |

Estimativa de API: ~1.200 chamadas Sabiá-4-thinking (aceitação ≈ 50 % observada
na rodada 2) — precisa da sua autorização antes de rodar.

### Fase 3 — Treino (½ dia GPU)

| # | Ação | Justificativa |
|---|---|---|
| 3.1 | Treinar sobre `train_curado_v3.jsonl` (≈ 2.000 exemplos), `EPOCHS=3`, `LORA_RANK=16` (manter); `MAX_SEQ_LENGTH=1024` basta para N=1 | LIMA: mais épocas em dados limpos > mais dados |
| 3.2 | Confirmar `train_on_responses_only` (perda só nos tokens do assistente) | padrão Unsloth para SFT de chat; evita aprender o prompt |
| 3.3 | Opcional, 2º treino só se 3.1 não passar: NEFTune (`neftune_noise_alpha=5`) | Jain et al., 2023: +ganho em instrução com datasets pequenos |
| 3.4 | Exportar para pasta nova `outputs/gguf_v3_gguf/` (nunca sobrescrever `gguf_gguf`) | rastreabilidade |
| 3.5 | Rodar `run_gate_planejado.sh` com `CAND=` novo, `BASE=` **modelo em produção** | veredito pareado |

Critério de saída: todos os P1–P7 + G11 aprovados; P8 ≤ +20 %; G1–G8 (N=1 sem
sufixo) sem regressão significativa (McNemar), como sinal de robustez.

### Fase 4 — Só se ainda reprovar: preferências (1 dia)

DPO ou ORPO com TRL sobre os pares da 2.5 (aceita = escolhida, rejeitada =
recusada), a partir do checkpoint SFT da Fase 3, β = 0,1, 1 época. Alvo
específico: gabarito inconsistente e dados ausentes. Só aqui porque DPO em
1,7 B com poucos pares pode degradar formato — a GBNF protege a estrutura, mas
a diversidade precisa ser reavaliada (G11).

### Fase 5 — Tempo no celular offline (paralelo, time do app)

Nada acima muda o modelo pesado; o que muda o tempo no aparelho é isto:

| # | Ação | Ganho esperado |
|---|---|---|
| 5.1 | llama.rn com o modelo **carregado uma vez** por sessão (não por chamada) | elimina 2–5 s de carga por questão |
| 5.2 | Cache do prefixo (1.4) | −20–35 % pré-fill |
| 5.3 | Streaming: mostrar a 1ª questão assim que validada; as N−1 seguem em segundo plano | percepção: espera = 1 questão, não N |
| 5.4 | Q4_K_M mantido; testar Q4_0 com kernels ARM (dotprod/i8mm) — costuma ser 1,3–1,8× mais rápido em Android recente; **só adotar se os gates passarem no artefato quantizado** | tempo |
| 5.5 | Medir P8 em 2 aparelhos reais (um de entrada, um médio) e fixar meta: ≤ 20 s/questão no de entrada | instrumento |
| 5.6 | Pré-gerar em ociosidade: quando o aluno abre a habilidade, gerar a próxima questão enquanto ele responde a atual | tempo percebido ≈ 0 |

Descartado: decodificação especulativa (precisa de 2º modelo em RAM),
temperatura maior (a regra do projeto), N questões numa chamada (evidência
do experimento A: 17 % não entregues, 24 % dependem de visual).

## 4. Ordem, esforço, custo

| Fase | Esforço | Custo externo | Decisão ao final |
|---|---|---|---|
| 0 Instrumento | ½ dia | 0 | — |
| 1 Guardas | 1 dia + 1,5 h de avaliação | 0 | **promover se passar** |
| 2 Dados v3 | 2–3 dias | ~1.200 chamadas Maritaca (autorizar) | — |
| 3 Treino | ½ dia GPU + 1,5 h | 0 | promover se passar |
| 4 DPO/ORPO | 1 dia | 0 | promover se passar |
| 5 Mobile | paralelo, time do app | 0 | meta de tempo no aparelho |

## 5. Riscos e como saber cedo

| Risco | Sinal | Mitigação |
|---|---|---|
| Guardas aumentam o tempo (mais regenerações) | P8 > +20 % | limitar a 1 retry por questão; degradar entregando o melhor candidato marcado |
| Dropout do sufixo reduz a aderência no modo C | P4 cai | fração 30 % → 15 % |
| Permutar alternativas quebra questões com "ordem" implícita ("todas as anteriores") | check_structure | não permutar quando alguma alternativa referencia outra |
| Falsos positivos novos do verificador | amostra manual (0.5) | manter a checagem manual de 40 itens em cada ciclo |
| DPO degrada diversidade | G11 | ficar no SFT; DPO só com G11 reavaliado |
| Quantização Q4_0 piora consistência | P3 no artefato | gates rodam sobre o GGUF final, nunca sobre o FP16 |

## 6. Pré-registro do próximo veredito

Fixado em 2026-09-30, antes da rodada de 32 prompts (decisão do responsável pelo
projeto, após simulação: tolerância 0 pp reprova o mesmo modelo por acaso em
~43 % das rodadas; 1 pp + teto em 7–12 %):
- Conjunto: 32 prompts × 3 seeds (≈ 456 questões por modelo), modo C, `MAX_TENTATIVAS_DIVERSIDADE=0`, código das Fases 0 e 1 (verificador revisado, guardas, permutação de letra).
- Bloqueantes: P1 ≥ 99 %; P2 = 100 %; **P3 inconsistentes ≤ baseline + 1 pp e ≤ 2 % absoluto** (a piora relativa pode ser absolvida por não-inferioridade pareada, o teto nunca); P4 ≥ baseline − 2 pp; P5 ≤ baseline + 1 pp; P6 ≥ baseline − 5 pp; P7; G11 (mesma tolerância de P3/P4 nas métricas de qualidade; diversity_score não pode piorar).
- Piloto: botão "reportar erro" no app; se as questões reportadas como erradas passarem de 2 % na primeira semana, volta o modelo anterior.
- Informativos: P8 tempo ≤ +20 % no PC; tempo no aparelho de entrada ≤ 20 s.
- Ganho exigido para promover: aderência **ou** inconsistentes **ou** cobertura melhor com significância.
