"""Formato de chat do Gemma 3 (1B) para o SFT (só stdlib; os testes rodam sem GPU).

O Gemma 3 usa `<start_of_turn>papel\\n ... <end_of_turn>\\n`, com o assistant renderizado
como `model`. O template funde o conteúdo do `system` como prefixo do primeiro turno de
`user` (não existe um turno de `system` próprio) e emite `<bos>` no início; o SFTTrainer
acrescenta outro `<bos>` ao tokenizar, então o texto de treino sai sem ele (mesma receita
do notebook oficial do Unsloth, `.removeprefix('<bos>')`).

Invariante que este módulo verifica: o texto de treino COMEÇA pelo prompt de geração
(system fundido no user + `<start_of_turn>model\\n`). Se o template de inferência
acrescentar algo ao prompt de geração que o treino não viu, o modelo é avaliado num
formato diferente do treinado; `verificar_template_gemma.py` compara o mesmo prompt com
o que o llama-server do .gguf renderiza.
"""

MARCADOR_USUARIO = "<start_of_turn>user\n"
MARCADOR_MODELO = "<start_of_turn>model\n"
FIM_TURNO = "<end_of_turn>"
BOS = "<bos>"


def remover_bos(texto, bos=BOS):
    return texto[len(bos):] if bos and texto.startswith(bos) else texto


def renderizar(tokenizer, messages, bos=BOS):
    """(texto_treino, prompt_geracao), ambos sem <bos>."""
    texto = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    return remover_bos(texto, bos), remover_bos(prompt, bos)


def problemas_formato(texto, prompt, resposta):
    """Lista de defeitos do exemplo renderizado; vazia = pronto para treinar."""
    erros = []
    if texto.startswith(BOS):
        erros.append("texto começa com <bos> (ficaria duplicado no SFTTrainer)")
    if texto.count(MARCADOR_USUARIO) != 1:
        erros.append(f"esperado 1 marcador de usuário, achei {texto.count(MARCADOR_USUARIO)}")
    if texto.count(MARCADOR_MODELO) != 1:
        erros.append(f"esperado 1 marcador de modelo, achei {texto.count(MARCADOR_MODELO)}")
    if not texto.startswith(prompt):
        erros.append("texto de treino não começa pelo prompt de geração (treino x inferência divergem)")
    elif not texto[len(prompt):].lstrip().startswith(resposta.strip()[:40]):
        erros.append("a resposta do assistant não vem logo após o marcador de modelo")
    if FIM_TURNO not in texto[len(prompt):]:
        erros.append("turno do modelo sem <end_of_turn> (o modelo não aprenderia a parar)")
    return erros


def diferenca(a, b, contexto=60):
    """Primeira posição onde a e b divergem, com um trecho de cada (para o relatório)."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    if i == len(a) == len(b):
        return None
    ini = max(0, i - contexto)
    return {"posicao": i, "a": a[ini:i + contexto], "b": b[ini:i + contexto]}
