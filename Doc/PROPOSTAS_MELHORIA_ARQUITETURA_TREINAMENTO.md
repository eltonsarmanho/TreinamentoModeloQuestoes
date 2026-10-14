# Auditoria de arquitetura e propostas de melhoria do treinamento

**Data da auditoria:** 14/10/2026  
**Escopo:** inspeção estática do repositório, artefatos locais de dados/treino/avaliação e documentação oficial/papers acessíveis em 14/10/2026.  
**Objetivo:** identificar riscos arquiteturais e oportunidades que possam melhorar a qualidade do gerador SAEB sem atribuir ganhos não demonstrados.

## Resumo executivo

A solução é um pipeline local de geração e avaliação de questões: SQLite como fonte de itens, exportação para JSONL conversacional, curadoria/sintético/destilação, QLoRA do Qwen3-1.7B, exportação GGUF e verificação na inferência. A base SQLite `DB/questoes.db` tem uma tabela principal `itens` (1.039 linhas) que combina enunciados, alternativas, gabarito, justificativas, habilidade/ano, proveniência e campos de imagem. O banco é fonte para preparação dos dados; o modelo treinado não consulta SQLite em tempo de inferência.

A configuração central do treino (`src/train.py`) é tecnicamente coerente com a restrição de GPU: base NF4 4-bit, LoRA r=16/alpha=32 nos módulos de atenção e MLP, batch efetivo 16, LR 2e-4, 3 épocas, cosine, warmup 5%, AdamW 8-bit, BF16 e perda mascarada no prompt. A literatura e a documentação atual dão suporte a vários desses valores como bons pontos de partida. Os dados do projeto, porém, mostram que **não se deve começar alterando hiperparâmetros**: duas execuções declaradamente iguais (v7b e v7d) tiveram resultados de gate diferentes; uma amostra humana de 20 itens por braço não encontrou ganho do v7b sobre o baseline; e v7c com sequência 2048 também piorou gates. Primeiro é preciso controlar a variância, fortalecer a validação e congelar o ambiente.

Há também uma divergência operacional de alta prioridade: o README conduz a `python src/train.py` sem argumentos, cujos padrões consomem `data/train.jsonl` (963 linhas presentes agora), enquanto o ciclo de treino mais recente usou explicitamente `data/train_curado_v7b.jsonl` (2.846 exemplos). As instruções e artefatos atuais podem levar alguém a treinar um modelo diferente do que os relatórios descrevem.

## Arquitetura observada

```text
DB/questoes.db (SQLite, tabela itens)
        │ extract_data.py / scripts de curadoria
        ├── data/train.jsonl (export padrão, 963 linhas no checkout atual)
        ├── data/train_curado_v7b.jsonl (candidata recente, 2.846 exemplos)
        └── data/val*.jsonl (conjuntos de avaliação de 30 exemplos cada)
                       │
                       ▼
    src/train.py: Qwen3-1.7B + QLoRA/Unsloth + TRL SFTTrainer
                       │
                       ├── outputs/checkpoints* (checkpoints por época)
                       └── outputs/lora* (adaptadores)
                                  │
                    evaluate.py / promover_checkpoint.py
                                  │
                     export_gguf.py → GGUF Q4_K_M
                                  │
       llama.cpp + GBNF + verificação/correção na inferência
```

O checkout contém scripts para extração, curadoria, destilação, geração sintética, auditoria de gabaritos, cobertura curricular, gates pareados e verificação da saída. Não encontrei código do aplicativo móvel ou um serviço de inferência/backend neste repositório; por isso a auditoria de runtime limita-se à integração descrita e aos artefatos de `llama.cpp`/GGUF aqui presentes.

### Dados e armazenamento

- `DB/questoes.db` contém 1.039 itens na tabela `itens`. O esquema mistura conteúdo da questão, quatro alternativas da fonte, gabarito, justificativas, metadados curriculares e colunas de imagem. A conversão para o contrato do gerador ocorre nos scripts Python.
- Os JSONL de treino são snapshots derivados, não tabelas consultadas no treino. Os campos `meta` guardam alguma proveniência, mas a representação varia entre gerações antigas e recentes.
- A extração padrão implementa split estratificado 90/10 e tem uma proteção contra sobreposição de `codigo_item` entre seus conjuntos. O conjunto congelado atual também não tem códigos em comum com `train_curado_v7b.jsonl` (verificação por identificador: 0/30).
- `data/val.jsonl`, `data/val_frozen_v1.jsonl` e `data/val_novos_v1.jsonl` têm apenas 30 exemplos cada. Isso é útil para smoke test e comparação histórica limitada, mas não oferece precisão suficiente para seleção fina de hiperparâmetros ou estimativa estreita de qualidade por habilidade.
- O relatório gerado `Doc/RELATORIO_BASE_v3.md` descreve `train_curado_v3`, não o estado v7b atual. Deve ser tratado como histórico, não como inventário atual.

### Modelo e treino

- Modelo base declarado: `unsloth/Qwen3-1.7B`; carregado em 4-bit e usado com `enable_thinking=False` ao formatar os exemplos.
- A adaptação cobre `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- `MAX_SEQ_LENGTH=1024`; o próprio histórico registra que mudar para 2048 não resolveu a regressão e que só 17 exemplos (0,6%) ultrapassam 1024 tokens.
- O script padrão recebe paths configuráveis para treino, validação, saída e checkpoints, mas não há manifesto de execução que congele hash do dataset, revisão do modelo/tokenizer, versões efetivamente instaladas e configuração completa no mesmo local.
- `requirements.txt` deixa `torch`, `unsloth` e dependências importantes sem versões exatas. Isso facilita instalação, mas pode alterar kernels/otimizadores e resultados entre execuções.

## Evidência empírica do projeto

1. **v7b não demonstrou ganho humano.** `ciclo_v7/SPEC.md` registra gate multiseed aprovado para v7b, porém amostra cega de 20 questões por braço com validade geral 35% no v7b versus 45% no baseline; correção matemática 60% versus 70%. A amostra tem baixa potência e não prova que o candidato seja inferior, mas não sustenta promoção.
2. **O resultado de treino não é reproduzível o bastante para comparar pequenas mudanças.** A repetição v7d da configuração v7b reprovou em gates que v7b havia passado, apesar de mesmo dataset, `SEED=42`, comprimento 1024 e hiperparâmetros declarados. O histórico atribui isso possivelmente a não determinismo dos kernels, sem isolar a causa.
3. **Aumentar sequência não é melhoria validada.** v7c com 2048 foi reprovado; v7d com 1024 também, portanto o efeito de `MAX_SEQ_LENGTH` continua confundido pela variância entre execuções. Manter 1024 é razoável por enquanto.
4. **A métrica de perda não basta.** Os três treinos selecionaram época 2, com `eval_loss` entre 0,67204 e 0,67356, mas os gates de geração variaram. A loss de next-token não mede diretamente validade pedagógica, correção matemática, autocontenção ou qualidade dos distratores.
5. **A validação humana é pequena.** n=20 por modelo dá intervalos largos. O próprio ciclo registra reprovações por erros não cobertos pelo checker atual, incluindo resolução correta sem alternativa correta, dependência de visual, premissas geométricas impossíveis e enunciados contraditórios.
6. **Risco operacional de treinar o corpus errado.** O README ainda instrui a executar `python src/train.py` como treino padrão. Sem `--train`, isso seleciona `data/train.jsonl`, que hoje tem 963 linhas; a candidata recente tem 2.846 exemplos e foi selecionada com `--train data/train_curado_v7b.jsonl`. O README também descreve 303 questões textuais, número histórico que não corresponde à contagem atual do arquivo.

## Avaliação dos hiperparâmetros atuais

| Parâmetro | Valor observado | Avaliação | Proposta |
|---|---:|---|---|
| Base | Qwen3-1.7B | Compatível com o alvo offline e com a capacidade de GPU declarada | Manter; registrar revisão imutável do Hub e hash/config/tokenizer por execução |
| Quantização de treino | 4-bit via Unsloth | Justificada por VRAM de 6 GB; QLoRA usa NF4 e outras técnicas de memória | Manter; confirmar e gravar `bnb_4bit_quant_type=nf4` e double quantization no manifesto, conforme a configuração efetiva da versão instalada |
| LoRA rank / alpha | 16 / 32 | Está dentro das faixas usuais, cobre atenção e MLP; não há evidência local de rank insuficiente | Manter no próximo ciclo controlado; depois comparar r=8,16,32 sem alterar outros fatores |
| Módulos alvo | atenção + MLP | Boa cobertura com custo pequeno; alinhado com Unsloth e achados recentes sobre aplicar LoRA em camadas lineares amplas | Manter |
| LR | 2e-4 | Plausível para r=16; fica próximo do valor ótimo observado no estudo de Schulman para aquele experimento (não é constante universal) | Não mudar sem sweep pareado. Rodar 1e-4, 2e-4, 3e-4 se houver base de avaliação robusta |
| Batch efetivo | 16 | Conservador e dentro do experimento de referência do repositório; grad accumulation reduz demanda de memória, mas não equivale em todos os aspectos a batch físico maior | Manter; evitar aumentar antes de estudar variância e medir throughput |
| Épocas | 3 | Faixa razoável para SFT pequeno; early stopping mede somente 3 pontos anuais | Manter para comparação inicial; avaliar checkpoint por época e considerar 2–4 somente com conjunto maior |
| Scheduler/warmup | cosine / 5% | Configuração comum e suportada pela documentação | Manter |
| Weight decay / dropout | 0,01 / 0 | Bons pontos de partida; Unsloth recomenda comumente dropout zero e alerta que dropout pode não regularizar de forma confiável em runs curtos | Manter; não adicionar NEFTune/dropout sem ablação isolada |
| Comprimento | 1024 | 17/2.846 amostras excedem esse limite, mas 2048 foi testado sem resultado favorável e a repetição 1024 também variou | Manter 1024; medir truncamento real e revisar esses exemplos antes de pagar o custo de sequência maior |
| Seleção do melhor | menor `eval_loss` na validação de 30 | Instrumento pequeno e descolado das métricas pedagógicas; não confiável para micro-otimização | Usar validação maior e conjunto de promoção separado; guardar checkpoints de todas as épocas |
| Seed/determinismo | 42, sem `full_determinism` | Seed fixa não garantiu resultado repetido no histórico | Separar seed de modelo e de amostragem; registrar ambiente; estimar distribuição em 3–5 seeds antes de atribuir ganho a um hiperparâmetro |

Os valores não devem ser considerados universalmente ótimos. O estudo LoRA Without Regret usa outros modelos, conjuntos e orçamento de treino; serve para informar o desenho do sweep, não para inferir diretamente que aumentar rank ou LR melhorará este gerador.

## Propostas de melhoria priorizadas

### P0 — Fixar o que realmente está sendo treinado

1. Atualizar instruções de treino para apontar explicitamente ao snapshot aprovado do corpus atual, ou exigir `--train` e `--val` para execução de produção. Marcar `data/train.jsonl` como export bruto/histórico caso continue necessário.
2. Criar manifesto por execução com: hash SHA-256 de cada JSONL e arquivo SQLite de origem, commit Git, versão/hash do modelo/tokenizer, versão de Python/CUDA/Torch/Unsloth/TRL/Transformers/PEFT/BitsAndBytes, todos os hiperparâmetros, semente, GPU, resumo do split, número de tokens truncados, melhor checkpoint e artefato GGUF derivado.
3. Fazer a promoção referenciar esse manifesto; bloquear promoção quando o hash dos dados ou a revisão do modelo não forem conhecidos.

**Aceite:** qualquer checkpoint pode ser ligado sem ambiguidade a dados, ambiente e configuração; o comando documentado treina o corpus que a documentação diz que treina.

### P1 — Medir a variância antes de tuning

1. Rodar 3–5 treinos independentes com a mesma configuração e corpus congelado. Variar `seed` de forma registrada; deixar data split e avaliação congelados. Não chamar os resultados de determinísticos.
2. Reportar média, dispersão e intervalo por seed para validade geral, correção matemática, questão autocontida/sem visual, dificuldade, alternativas plausíveis e taxa de geração corrigida/rejeitada.
3. Depois da linha de base de variância, testar um fator de cada vez: primeiro LR (1e-4/2e-4/3e-4), depois rank (8/16/32), mantendo comprimento e epochs. Evitar grid de combinações na GPU limitada.

**Aceite:** efeito de uma alteração excede a variabilidade entre seeds e se repete no GGUF final; nenhuma decisão vem de uma única execução.

### P2 — Reforçar a avaliação independente

1. Manter o `val` congelado atual como comparação histórica, mas criar uma avaliação nova e não usada em curadoria, com pelo menos 100 itens por braço humano como meta inicial e estratificação por ano/habilidade/dificuldade/origem.
2. Aumentar prompts/itens verificáveis automaticamente e reportar denominador verificável em todas as métricas; não apresentar “100% consistente” sem informar `n`.
3. Separar três camadas: validação de treino para escolher checkpoint; benchmark de promoção para comparar ao baseline; amostra humana cega para validade pedagógica. Não reutilizar itens/saídas da avaliação humana em curadoria.
4. Calibrar o juiz automático contra as anotações humanas existentes, medir sensibilidade/especificidade e analisar divergências. O juiz não substitui a amostra; serve para triagem e escala.
5. Incluir verificações estruturais e semânticas: resultado da resolução aparece em alguma alternativa, unidade compatível, todas as alternativas distintas após normalização, questão sem imagem/tabela omitida, coerência do enunciado e da habilidade. Marcar “não verificável” separadamente de “aprovado”.

**Aceite:** relatório com métricas, denominadores, intervalos de confiança e comparação pareada por item; sem vazamento por `codigo_item` ou near-duplicate entre treino e avaliação.

### P3 — Melhorar proveniência e modelo de dados

1. Tratar SQLite como fonte canônica e cada JSONL como artefato derivado versionado. Criar um manifesto do dataset com versão de schema, data da extração, critérios de exclusão, IDs, origem/juízes/edições humanas, hashes e contagens.
2. Tornar proveniência obrigatória no schema dos exemplos. Hoje o campo `meta` é heterogêneo (inclusive chaves ausentes nos exemplos de reforço); normalizar `codigo_item`, `origem`, `origem_id`, `ano`, `habilidade`, `dificuldade`, versão da curadoria e `status_verificacao`.
3. Adicionar tabela de auditoria/edições append-only para correções humanas, sem apagar o valor original; registrar autor, data, motivo e versão. Isso é especialmente importante porque a v7 contém decisões humanas que divergem do registro original do banco.
4. Formalizar migrações/versionamento do esquema do SQLite antes de mudanças de colunas; manter imagens/blobs como referências externas quando possível, preservando hash e URI em vez de duplicar binários nos snapshots.

**Aceite:** é possível reconstruir cada exemplo treinado até o item de origem e a transformação aplicada; correções humanas são auditáveis e reversíveis.

### P4 — Integrar o formato conversacional à API corrente

A perda só nas respostas é conceitualmente adequada. A documentação atual do TRL oferece `assistant_only_loss=True` em `SFTConfig` e declara suporte/patch para templates conhecidos como Qwen3. O código atual usa `train_on_responses_only` do Unsloth. Recomendo avaliar a API nativa em uma branch experimental, mas só migrar após provar equivalência da máscara de tokens para o template exato do projeto e comparar o mesmo corpus. Evitar atualizar o TRL e trocar o método de loss na mesma experiência.

**Aceite:** inspeção de um batch confirma que todos os tokens de `user/system` estão mascarados e todos os tokens de resposta desejados estão supervisionados; o resultado passa a comparação de regressão.

### P5 — Aumentar a verificabilidade matemática no produto

O contrato de saída guarda a letra e uma explicação livre; o relatório do projeto reconhece cobertura limitada do regex de `check_consistency`. Sem alterar o contrato do aplicativo, a etapa de validação pode converter a questão para uma representação interna estruturada (expressão/resultado/unidade) e validar alternativas antes de aceitar a saída. Para itens conceituais ou textuais, manter revisão semântica separada e o estado “não verificável”. Se o contrato puder evoluir no futuro, considerar campo interno de resposta normalizada, sem expô-lo ao aluno.

**Aceite:** medir cobertura, falso positivo e falso negativo do verificador contra um conjunto humano rotulado antes de usá-lo como gate ou recompensa de treino.

### P6 — Curadoria: balancear o currículo com evidência, não só contagem

O corpus recente contém 2.846 exemplos e agrega exemplos reais, injetados e reforço sintético. A documentação v3 já apontava lacunas por subtema, concentração em “outros” e mais de metade dos gabaritos sem verificação aritmética; o ciclo v7 mostra que a curadoria pode melhorar a distribuição de gabaritos e ainda assim não superar o baseline em revisão humana. Assim:

- Priorizar itens com proveniência e verificação fortes, diversidade semântica real e subtemas ausentes.
- Monitorar proporções de real/sintético/destilado/injetado por habilidade, dificuldade e subtema; impedir que volume gerado substitua âncoras reais sem motivo documentado.
- Executar deduplicação semântica e checagem de contaminação sobre texto normalizado e metadados, não só ID.
- Avaliar a regra “E = Nenhuma das alternativas anteriores”: evidência v7 indica que removê-la reduz viés, mas pode deslocar para geração de distratores incorretos. A métrica-alvo deve ser qualidade dos distratores e acerto, não a frequência isolada de NDA.

## Plano de execução recomendado

| Ordem | Ação | Custo relativo | Risco reduzido |
|---|---|---:|---|
| 1 | Corrigir README/CLI para garantir corpus e split corretos; gerar manifesto | Baixo | Treino acidental com snapshot errado e resultados irreproduzíveis |
| 2 | Congelar ambiente/base/dados e fazer 3–5 seeds com configuração atual | Médio | Confundir ruído de execução com efeito de hiperparâmetro |
| 3 | Reforçar conjunto humano/avaliação independente e denominadores | Médio/alto | Promover regressões pedagógicas invisíveis aos gates |
| 4 | Fazer sweep pequeno de LR, depois rank, mantendo todo o resto fixo | Médio | Otimização sem evidência no domínio-alvo |
| 5 | Normalizar proveniência e adicionar trilha de auditoria de correções | Médio | Perda de rastreabilidade entre SQLite, JSONL e checkpoint |
| 6 | Avaliar a migração para `assistant_only_loss` e melhorar verificador estruturado | Médio | Fragilidade de dependência e baixa cobertura matemática |

## Decisões imediatas

- **Manter** QLoRA, rank 16/alpha 32, LR 2e-4, batch efetivo 16, 3 épocas, módulos atuais e sequência 1024 como baseline experimental.
- **Não** elevar `MAX_SEQ_LENGTH` apenas por causa dos 17 exemplos longos; a evidência v7c/v7d não isola benefício e custo.
- **Não** fazer busca de hiperparâmetros nem promover apenas pelo `eval_loss` de 30 itens.
- **Priorizar** alinhamento do comando padrão com o corpus mais recente, manifesto reproduzível, execução multiseed e avaliação humana mais ampla.

## Limitações desta auditoria

- Foi feita inspeção estática de arquivos e relatórios existentes. Não rodei treinamento, inferência, aplicação dos gates, testes ou migrações.
- Relatórios antigos foram identificados pelo nome/ciclo/data e não tratados como resultados atuais.
- O código-fonte do app móvel não está neste repositório; recomendações de integração com mobile são limitadas às interfaces de exportação e aos relatórios locais.
- Recomendações externas sustentam princípios e desenhos de experimento; não garantem ganho no modelo/dataset SAEB deste projeto.

## Fontes externas

Consultadas em 14/10/2026; priorizadas fontes primárias e documentação oficial.

1. Dettmers et al., **QLoRA: Efficient Finetuning of Quantized LLMs**, NeurIPS 2023. NF4, double quantization e paged optimizers: [paper](https://arxiv.org/abs/2305.14314).
2. Hugging Face, **TRL SFTTrainer**, documentação consultada em outubro de 2026. `assistant_only_loss`, packing e máscaras por template: [documentação atual](https://huggingface.co/docs/trl/main/sft_trainer).
3. Schulman et al./Thinking Machines Lab, **LoRA Without Regret**, 29/09/2025. Estudos controlados de rank, LR, batch e camadas alvo; resultados dependem do regime experimental: [artigo](https://thinkingmachines.ai/blog/lora/).
4. Unsloth, **LoRA fine-tuning Hyperparameters Guide**, documentação corrente. Faixas recomendadas para rank/alpha, módulos, épocas e dropout: [guia](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
5. Hugging Face, **Transformers Trainer**, documentação corrente. `seed`, `data_seed` e `full_determinism`, incluindo custo de desempenho da determinização: [documentação](https://huggingface.co/docs/transformers/main_classes/trainer).
6. Qwen, **Qwen3-1.7B model card**. Modos thinking/non-thinking e recomendações de amostragem para modo non-thinking: [model card](https://huggingface.co/Qwen/Qwen3-1.7B).
7. Cobbe et al., **Training Verifiers to Solve Math Word Problems**, 2021. Amostragem de candidatos e seleção por verificador em GSM8K: [paper](https://arxiv.org/abs/2110.14168).
