"""Árbitro de OUTRA FAMÍLIA de modelo (Google Gemini) para decisões de baixo volume.

Por que existe (H4, decisão do usuário de 2026-10-01): o validador e o revisor
são da mesma família (sabia-4-thinking) e erram JUNTOS. No idx 881 ("começou
numa terça e durou 10 dias") os dois contaram "sexta" e um item correto ia
sair da v3 como "erro matemático confirmado"; no idx 1348 (5/11 com gabarito
"0,45") os dois aprovaram juntos. A concordância deles não é um sinal
independente. Antes de QUALQUER remoção automática por erro matemático (D2), um
modelo de outra família resolve o item às cegas: o item só sai se ele também
encontrar erro matemático; se discordar, vai para revisão humana.

Uso restrito a tarefas de BAIXO volume que exigem raciocínio (árbitro da D2,
desempate de dificuldade na recalibração). Orçamento próprio e separado do da
Maritaca (Orcamento do agentes_questoes, outro teto, outro log de uso).

Modelo: o usuário pediu "gemini-2.0-flash-thinking-exp-01-21". Se a API não o
oferece mais, NÃO trocamos em silêncio: `escolher_modelo` lista os modelos da
própria API e escolhe o mais próximo (flash com thinking); a troca fica
registrada em MODELO_ARBITRO/MOTIVO_TROCA_MODELO abaixo e no relatório.

Cliente leve (requests, já no venv) e INJETÁVEL: `http` é qualquer objeto com
get/post no formato do requests; os testes usam um falso, sem rede. A chave
(GOOGLE_API_KEY, carregada do .env pelo distill_teacher como o resto do
projeto) vai SÓ no cabeçalho x-goog-api-key: nunca na URL (a mensagem de
HTTPError do requests traz a URL), nunca em log, repr ou registro.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import agentes_questoes as aq

ROOT = Path(__file__).resolve().parent.parent
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
LOG_USO_GEMINI = ROOT / "outputs" / "agentes" / "uso_api_gemini.jsonl"
CACHE_ARBITRO = ROOT / "outputs" / "agentes" / "arbitro_cache.jsonl"
ARBITRAGEM_D2 = ROOT / "outputs" / "agentes" / "arbitragem_d2.jsonl"
ESCOLHA_MODELO = ROOT / "outputs" / "agentes" / "arbitro_modelo.json"

# Modelo pedido pelo usuário (H4). Ver MODELO_ARBITRO para o efetivamente usado.
MODELO_PEDIDO = "gemini-2.0-flash-thinking-exp-01-21"
# Modelo EFETIVO (configuração). TROCA REGISTRADA em 2026-10-01: a listagem da
# própria API (1 chamada, GET /v1beta/models; lista gravada em
# outputs/agentes/arbitro_modelo.json) NÃO tem mais o modelo pedido. Pelo
# critério de escolher_modelo() — flash com thinking, estável, versão mais
# próxima da pedida — o substituto é o gemini-2.5-flash. Há flash mais novos
# com thinking (gemini-3.5/3.6/3.7/3.8-flash); usar um deles é decisão do
# usuário (mude esta constante ou passe --modelo-arbitro).
MODELO_ARBITRO = "gemini-2.5-flash"
MOTIVO_TROCA_MODELO = ("gemini-2.0-flash-thinking-exp-01-21 não está mais na lista da API (2026-10-01); "
                       "substituto mais próximo: gemini-2.5-flash (flash, thinking, estável, versão mais próxima)")

# Temperatura 0 (árbitro, não gerador). Modelos thinking gastam o teto de
# saída também no raciocínio: com teto baixo o JSON sai truncado.
TEMPERATURA = 0.0
MAX_TOKENS = 8192
TETO_PADRAO_CHAMADAS = 40  # orçamento separado e baixo (H4)
# 2º revisor: o Gemini "pensa" antes de responder e esses tokens contam no limite; com 8.192,
# 4 de 30 respostas vieram cortadas (MAX_TOKENS) na calibração de 2026-10-02 e virariam
# reprovação falsa (fail-closed). Só o revisor2 usa o limite maior (a chave do cache do
# árbitro, que inclui MAX_TOKENS, não muda).
MAX_TOKENS_REVISOR2 = 24576

# Acréscimo ao prompt do revisor SÓ para o Gemini (o do sabiá não muda). Na calibração de
# 2026-10-02 ele reprovou por "ano_inadequado" triângulos do 9º ano H17 corretos ("escaleno é
# de anos anteriores; no 9º se espera algo mais complexo"): confundia item FÁCIL com item de
# ANO INADEQUADO, apesar de a própria habilidade da matriz incluir esse conteúdo.
REVISOR2_ADDENDUM = """

REGRAS ADICIONAIS DO 2º REVISOR (valem acima de qualquer outra leitura dos critérios C1 e C2):
- O conteúdo descrito na habilidade informada (Matriz de Referência do SAEB) É o conteúdo esperado para o ano pedido, mesmo que o tema já tenha sido introduzido em anos anteriores: a Matriz retoma e aprofunda conteúdos. Não reprove por "o ano esperaria algo mais complexo".
- C2 (ano_inadequado) só é false quando o item exige conteúdo, vocabulário ou números que NÃO fazem parte da habilidade informada (ex.: trigonometria numa habilidade de classificação) ou, nos anos iniciais, conteúdo de ano posterior.
- Item simples não é item inadequado: o nível baixo vai SOMENTE em "dificuldade_real" (C3), nunca em C1 ou C2.
- C5 (enunciado ambíguo) só é false se uma leitura razoável do enunciado levar a outra resposta; frase redundante ou desnecessária é sugestão, não veto.
- Todo o rigor matemático (premissa possível, dados suficientes, UMA única alternativa verdadeira, distratores estritamente falsos) continua valendo integralmente.
"""
VERSAO_PROMPT_REVISOR2 = hashlib.sha1(REVISOR2_ADDENDUM.encode("utf-8")).hexdigest()[:12]

# O árbitro usa o MESMO prompt do validador cego (regras da BNCC, D5, H2 e
# formato de saída que o código já interpreta), numa permutação própria das
# alternativas (sal "arbitro"): só a família do modelo muda.
ARBITRO_SISTEMA = aq.VALIDADOR_SISTEMA
ARBITRO_USUARIO = aq.VALIDADOR_USUARIO
VERSAO_PROMPT_ARBITRO = hashlib.sha1((ARBITRO_SISTEMA + "\x1e" + ARBITRO_USUARIO).encode("utf-8")).hexdigest()[:12]

# DESEMPATE DE DIFICULDADE (passo 2 da recalibração, 2026-10-01): o Gemini
# julga a dificuldade_real pela MESMA rubrica absoluta da C3 do revisor (texto
# extraído do REVISOR_SISTEMA, não copiado: se a rubrica mudar lá, muda aqui e
# a versão do prompt muda junto). Sem gabarito, sem resolução do autor e sem a
# dificuldade pedida: a âncora da dificuldade pedida é justamente o viés que o
# P3 tirou do revisor (piloto 2: 47% dos Moderado/Difícil "rebaixados").
def _rubrica_c3(sistema=None):
    s = sistema if sistema is not None else aq.REVISOR_SISTEMA
    ini = s.index("RUBRICA ABSOLUTA.") + len("RUBRICA ABSOLUTA.")
    fim = s.index("C3 é sempre true")
    return s[ini:fim].strip()


ARBITRO_DIFICULDADE_SISTEMA = (
    "Você é especialista em avaliação educacional e elaborador experiente de itens do SAEB (INEP), com prática em "
    "sala de aula do Ensino Fundamental. Sua única tarefa é dizer a DIFICULDADE de um item de Matemática para um "
    "aluno do ano informado. Você não recebe o gabarito nem a dificuldade que o autor pretendia: resolva o item "
    "você mesmo e classifique por esta RUBRICA ABSOLUTA.\n" + _rubrica_c3() + "\n"
    "Responda APENAS com um JSON numa linha, nesta ordem de campos: "
    '{"resolucao": str (até 4 frases), "etapas": int, "exigencias": [str], '
    '"dificuldade_real": "Fácil|Moderado|Difícil"}')
ARBITRO_DIFICULDADE_USUARIO = """Ano: {ano} ano
Habilidade: {habilidade} — {descricao}

Enunciado: {enunciado}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}"""
VERSAO_PROMPT_DIFICULDADE = hashlib.sha1((ARBITRO_DIFICULDADE_SISTEMA + "\x1e" + ARBITRO_DIFICULDADE_USUARIO)
                                         .encode("utf-8")).hexdigest()[:12]


def interpretar_dificuldade(texto):
    """{dificuldade_real, etapas, exigencias, resolucao} ou None (malformado:
    sem objeto JSON final ou dificuldade fora das três)."""
    o = aq.extrair_json(texto, ("dificuldade_real",), estrito=True)
    if not isinstance(o, dict):
        return None
    dif = aq.dificuldade_valida(o.get("dificuldade_real"))
    if dif is None:
        return None
    etapas = o.get("etapas")
    return {"dificuldade_real": dif, "etapas": etapas if isinstance(etapas, int) else None,
            "exigencias": [str(x)[:120] for x in (o.get("exigencias") or []) if isinstance(x, str)][:6],
            "resolucao": str(o.get("resolucao") or "")[:600]}


# papel -> função que diz se a resposta é interpretável (cache só guarda e só
# devolve resposta interpretável, P4).
_INTERPRETAVEL = {"arbitro": lambda t: aq.resposta_interpretavel("validador", t),
                  "arbitro_dificuldade": lambda t: isinstance(t, str) and interpretar_dificuldade(t) is not None,
                  "revisor2": lambda t: aq.resposta_interpretavel("revisor", t)}


# Partes do nome que tiram um modelo da lista de candidatos a árbitro (não
# geram texto de raciocínio ou são variantes reduzidas).
_EXCLUIR = re.compile(r"lite|image|tts|audio|live|embedding|aqa|vision|robotics|learnlm|native", re.I)


class ErroGemini(RuntimeError):
    """Erro HTTP do Gemini com o status (para aq._retentavel) e sem a chave."""

    def __init__(self, status_code, mensagem):
        super().__init__(f"HTTP {status_code}: {mensagem}")
        self.response = SimpleNamespace(status_code=status_code)


class ClienteGemini:
    """generateContent / models do Gemini via REST. Interface de chat igual à
    dos outros clientes (chat_completion -> .choices[0].message.content,
    .usage, .choices[0].finish_reason)."""

    def __init__(self, modelo=MODELO_ARBITRO, api_key=None, timeout=240, http=None):
        self.modelo = modelo
        self._chave = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        if http is None:
            import requests as http  # noqa: PLC0415 - leve, já no venv
        self.http = http

    def __repr__(self):  # nunca expor a chave num traceback/log
        return f"ClienteGemini(model={self.modelo!r})"

    def _cabecalhos(self):
        if not self._chave:
            raise SystemExit("GOOGLE_API_KEY ausente no .env")
        return {"x-goog-api-key": self._chave, "Content-Type": "application/json"}

    @staticmethod
    def _checar(r):
        st = getattr(r, "status_code", 200)
        if st >= 400:
            try:
                msg = (r.json().get("error") or {}).get("message", "")
            except Exception:  # noqa: BLE001
                msg = ""
            raise ErroGemini(st, aq._sanitizar(msg, 200))

    def listar_modelos(self):
        """Modelos da API (uma chamada; pageSize alto para caber numa página)."""
        r = self.http.get(f"{GEMINI_URL}/models", headers=self._cabecalhos(), params={"pageSize": 1000},
                          timeout=self.timeout)
        self._checar(r)
        return r.json().get("models") or []

    def chat_completion(self, messages, max_tokens, temperature):
        sistema = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        contents = [{"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m.get("role") != "system"]
        corpo = {"contents": contents,
                 "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if sistema:
            corpo["systemInstruction"] = {"parts": [{"text": sistema}]}
        r = self.http.post(f"{GEMINI_URL}/models/{self.modelo}:generateContent", headers=self._cabecalhos(),
                           json=corpo, timeout=self.timeout)
        self._checar(r)
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        partes = (cand.get("content") or {}).get("parts") or []
        # partes de "pensamento" (thought=true), quando vierem, não são a resposta
        texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
        um = data.get("usageMetadata") or {}
        saida = int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0)
        usage = (SimpleNamespace(prompt_tokens=um.get("promptTokenCount"), completion_tokens=saida)
                 if um else None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto),
                                                        finish_reason=cand.get("finishReason"))],
                               usage=usage, thoughts_tokens=um.get("thoughtsTokenCount"))


def _versao(nome):
    """(2, 5) de 'gemini-2.5-flash'; (0,) se não houver número."""
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", nome)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0,)


def escolher_modelo(modelos, pedido=MODELO_PEDIDO):
    """(nome, motivo) do modelo a usar como árbitro.

    1. o PEDIDO, se a API o lista com generateContent;
    2. senão, o mais próximo: "flash" com thinking ("thinking" no nome ou o
       campo thinking=true da API), sem variantes lite/imagem/áudio, estável
       (sem preview/exp/latest) e com a versão MAIS PRÓXIMA da pedida — não a
       mais nova: o usuário escolheu um flash da geração 2.0, e o substituto
       direto é o 2.5-flash, não o 3.x (que fica como opção explícita);
    3. sem candidato, None (o chamador para; não troca de família)."""
    disp = {}
    for m in modelos or []:
        nome = str(m.get("name", "")).split("/")[-1]
        if nome and "generateContent" in (m.get("supportedGenerationMethods") or []):
            disp[nome] = m
    if pedido in disp:
        return pedido, "pedido_disponivel"
    cand = [n for n, m in disp.items()
            if "flash" in n and ("thinking" in n or m.get("thinking") is True) and not _EXCLUIR.search(n)]
    if not cand:
        return None, "nenhum_flash_com_thinking"
    estavel = [n for n in cand if not re.search(r"preview|exp|latest|\d{2}-\d{2}$", n)]
    pool = estavel or cand
    alvo = _versao(pedido)

    def distancia(n):
        v = _versao(n)
        return abs((v[0] + (v[1] if len(v) > 1 else 0) / 10) - (alvo[0] + (alvo[1] if len(alvo) > 1 else 0) / 10))

    pool.sort(key=lambda n: (distancia(n), len(n), n))
    return pool[0], (f"pedido_indisponivel ({pedido} não está na lista da API); mais próximo flash com thinking, "
                     f"estável e de versão mais próxima: {pool[0]}")


class Arbitro:
    """Resolve o item ÀS CEGAS com o Gemini (mesmo contrato do validador fase 1).

    Orçamento separado (aq.Orcamento próprio), log de uso próprio e cache só de
    respostas interpretáveis e não truncadas (P4)."""

    def __init__(self, cliente, modelo=None, orcamento=None, log_uso=LOG_USO_GEMINI, cache_path=CACHE_ARBITRO,
                 tentativas=2, backoff=4.0, dormir=time.sleep):
        self.cliente = cliente
        self.modelo = modelo or getattr(cliente, "modelo", MODELO_ARBITRO)
        self.orcamento = orcamento or aq.Orcamento(TETO_PADRAO_CHAMADAS)
        self.log_uso = Path(log_uso) if log_uso else None
        self.cache_path = Path(cache_path) if cache_path else None
        self.tentativas = max(1, int(tentativas))
        self.backoff = backoff
        self.dormir = dormir
        self.cache = {}
        self.hits = 0
        if self.cache_path and self.cache_path.exists():
            for linha in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                ok = _INTERPRETAVEL.get(r.get("papel") or "arbitro")
                if r.get("chave") and ok is not None and ok(r.get("texto")):
                    self.cache[r["chave"]] = r

    def _chave(self, mensagens, papel="arbitro"):
        # "arbitro" mantém a chave antiga (o cache já pago continua valendo).
        mt = MAX_TOKENS_REVISOR2 if papel == "revisor2" else MAX_TOKENS
        base = json.dumps([papel, self.modelo, mt, TEMPERATURA, mensagens], ensure_ascii=False,
                          sort_keys=True)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _log(self, reg):
        if self.log_uso:
            with aq.abrir_para_gravar(self.log_uso, "a") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    def chamar(self, mensagens, fase="arbitro_cego", papel="arbitro"):
        chave = self._chave(mensagens, papel)
        if chave in self.cache:
            self.hits += 1
            return self.cache[chave]["texto"], {"cache": True}
        ultimo = None
        for tentativa in range(1, self.tentativas + 1):
            self.orcamento.reservar(papel)
            t0 = time.monotonic()
            reg = {"ts": aq.agora_iso(), "agente": papel, "fase": fase, "modelo": self.modelo,
                   "tentativa": tentativa, "provedor": "google"}
            try:
                resp = self.cliente.chat_completion(
                    messages=mensagens, max_tokens=MAX_TOKENS_REVISOR2 if papel == "revisor2" else MAX_TOKENS,
                    temperature=TEMPERATURA)
                texto = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - rede/quota/provedor
                ultimo = aq._sanitizar(f"{type(exc).__name__}: {exc}")
                reg.update(ok=False, erro=ultimo, latencia_s=round(time.monotonic() - t0, 2), prompt_tokens=0,
                           completion_tokens=0, estimado=True)
                self._log(reg)
                if tentativa < self.tentativas and aq._retentavel(exc):
                    self.dormir(self.backoff * (2 ** (tentativa - 1)))
                    continue
                return None, {"erro": ultimo}
            usage = aq._ler_usage(resp)
            estimado = usage is None
            if estimado:
                usage = (sum(aq._estimar_tokens(m.get("content", "")) for m in mensagens), aq._estimar_tokens(texto))
            self.orcamento.registrar_tokens(papel, usage[0], usage[1], estimado)
            fim = getattr(resp.choices[0], "finish_reason", None)
            reg.update(ok=True, latencia_s=round(time.monotonic() - t0, 2), prompt_tokens=usage[0],
                       completion_tokens=usage[1], estimado=estimado, finish_reason=fim,
                       thoughts_tokens=getattr(resp, "thoughts_tokens", None))
            self._log(reg)
            info = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "finish_reason": fim}
            if (self.cache_path and not aq.resposta_truncada(info)
                    and _INTERPRETAVEL[papel](texto)):
                r = {"chave": chave, "papel": papel, "modelo": self.modelo, "texto": texto,
                     "prompt_tokens": usage[0], "completion_tokens": usage[1], "ts": aq.agora_iso(),
                     "versao_prompt_arbitro": (VERSAO_PROMPT_ARBITRO if papel == "arbitro"
                                               else aq.VERSAO_PROMPTS if papel == "revisor2"
                                               else VERSAO_PROMPT_DIFICULDADE)}
                self.cache[chave] = r
                with aq.abrir_para_gravar(self.cache_path, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return texto, info
        return None, {"erro": ultimo}

    def resolver_cego(self, questao):
        """{veredito, avaliado, resposta_calculada (letra ORIGINAL), status,
        problemas, confianca, resolucao, modelo}. Sem gabarito nem resolução do
        autor; a comparação com o gabarito é feita aqui. Falha = avaliado False
        (nunca vira "concorda" nem "discorda")."""
        alts = questao.get("alternativas") or {}
        gab = questao.get("resposta_correta")
        perm = aq.permutacao(questao, sal="arbitro")
        user = ARBITRO_USUARIO.format(enunciado=questao.get("enunciado", ""), **{L: alts.get(perm[L], "")
                                                                                for L in aq.LETRAS})
        msgs = [{"role": "system", "content": ARBITRO_SISTEMA}, {"role": "user", "content": user}]
        texto, info = self.chamar(msgs)
        base = {"modelo": self.modelo, "permutacao": perm, "versao_prompt_arbitro": VERSAO_PROMPT_ARBITRO}
        if texto is None:
            return dict(base, veredito=False, avaliado=False, erro="erro_api", resposta_calculada=None,
                        problemas=[{"codigo": "erro_api", "detalhe": str(info.get("erro"))[:200]}], confianca=None)
        f1 = aq.interpretar_fase1(texto)
        if f1 is None or aq.resposta_truncada(info):
            return dict(base, veredito=False, avaliado=False, erro="veredito_malformado", resposta_calculada=None,
                        problemas=[{"codigo": "veredito_malformado", "detalhe": "arbitro"}], confianca=None,
                        bruto=aq._sanitizar(texto, 400))
        status = {perm[L]: f1["status"][L] for L in aq.LETRAS}
        resp_exib = aq.resposta_unica(f1["status"])
        resp = perm[resp_exib] if resp_exib else None
        problemas = list(f1["problemas"])
        if not f1["premissa_possivel"] and not any(p["codigo"] == "premissa_impossivel" for p in problemas):
            problemas.append({"codigo": "premissa_impossivel", "detalhe": "premissa_possivel=false"})
        equivalentes = [[perm[a], perm[b]] for a, b in f1["equivalentes"]]
        if equivalentes and not any(p["codigo"] == "alternativas_equivalentes" for p in problemas):
            problemas.append({"codigo": "alternativas_equivalentes", "detalhe": str(equivalentes)})
        if f1["dependencia_visual"] and not any(p["codigo"] == "dependencia_visual" for p in problemas):
            problemas.append({"codigo": "dependencia_visual", "detalhe": "dependencia_visual=true"})
        gerais = sorted(L for L in aq.LETRAS if status[L] == "G")
        if gerais and not any(p["codigo"] == "distrator_verdadeiro" for p in problemas):
            problemas.append({"codigo": "distrator_verdadeiro", "detalhe": f"classe mais geral também vale: {gerais}"})
        if resp != gab:
            problemas.append({"codigo": "gabarito_sem_resposta_unica" if resp is None else "gabarito_errado",
                              "detalhe": f"arbitro={resp} gabarito={gab}"})
        return dict(base, veredito=bool(resp == gab and not problemas), avaliado=True, resposta_calculada=resp,
                    status=status, problemas=problemas, confianca=f1["confianca"], resolucao=f1["resolucao"],
                    cache=bool(info.get("cache")))


    def revisar_cego(self, agentes, questao, ano, habilidade, descricao, subtema=None):
        """2º REVISOR (2026-10-02): o prompt e a interpretação do revisor da
        Maritaca, enviados ao Gemini numa permutação própria (sal "revisor2").
        Mesmo contrato de Agentes.revisar. Orçamento, log e cache do Gemini."""
        return agentes.revisar(
            questao, ano, habilidade, descricao, subtema, None, sal="revisor2", sistema_extra=REVISOR2_ADDENDUM,
            chamador=lambda msgs: self.chamar(msgs, fase="revisao_cega", papel="revisor2"))

    def julgar_dificuldade(self, questao, ano, habilidade, descricao=""):
        """Dificuldade real pela rubrica absoluta (desempate da D3). Contrato:
        {avaliado, dificuldade_real|None, etapas, exigencias, resolucao, modelo,
        versao_prompt_dificuldade, cache}. Falha (erro de API, JSON malformado
        ou resposta truncada) = avaliado False: nunca conta como "Fácil"."""
        alts = questao.get("alternativas") or {}
        user = ARBITRO_DIFICULDADE_USUARIO.format(ano=ano, habilidade=habilidade, descricao=descricao or "",
                                                  enunciado=questao.get("enunciado", ""),
                                                  **{L: alts.get(L, "") for L in aq.LETRAS})
        msgs = [{"role": "system", "content": ARBITRO_DIFICULDADE_SISTEMA}, {"role": "user", "content": user}]
        texto, info = self.chamar(msgs, fase="dificuldade", papel="arbitro_dificuldade")
        base = {"modelo": self.modelo, "versao_prompt_dificuldade": VERSAO_PROMPT_DIFICULDADE,
                "cache": bool(info.get("cache"))}
        if texto is None:
            return dict(base, avaliado=False, dificuldade_real=None, erro="erro_api")
        d = interpretar_dificuldade(texto)
        if d is None or aq.resposta_truncada(info):
            return dict(base, avaliado=False, dificuldade_real=None, erro="veredito_malformado",
                        bruto=aq._sanitizar(texto, 300))
        return dict(base, avaliado=True, **d)


def montar_arbitro(dry_run=False, max_chamadas=TETO_PADRAO_CHAMADAS, log_uso=LOG_USO_GEMINI, cache_path=CACHE_ARBITRO,
                   modelo=None):
    """Árbitro real (Gemini) ou, em dry-run, com o cliente simulado do projeto
    (que reconhece o prompt do validador e não usa rede)."""
    modelo = modelo or MODELO_ARBITRO
    if dry_run:
        cli = aq.ClienteSimulado(erro_a_cada=0)
        return Arbitro(cli, f"simulado:{modelo}", aq.Orcamento(max_chamadas), log_uso=log_uso, cache_path=None)
    import distill_teacher  # noqa: F401,PLC0415 - carrega o .env (sem imprimir nada dele)
    return Arbitro(ClienteGemini(modelo), modelo, aq.Orcamento(max_chamadas), log_uso=log_uso, cache_path=cache_path)


LIMITE_FALHAS_GEMINI = 3  # falhas de API SEGUIDAS (cada uma já com retry) para dar o Gemini como indisponível


def ligar_segundo_revisor(agentes, dry_run=False, max_chamadas=TETO_PADRAO_CHAMADAS, log_uso=None,
                          cache_path=None, modelo=None, limite_falhas=LIMITE_FALHAS_GEMINI, aviso=print):
    """Liga o Gemini como 2º revisor em `agentes` (orçamento PRÓPRIO, separado
    da Maritaca). Devolve o Arbitro usado, para o chamador ler o uso.

    DEGRADAÇÃO (2026-10-02, pedido do usuário: o Gemini tem limite de gasto e a
    chamada falha ao ultrapassá-lo; o fluxo não pode acabar): falha de API
    (cota, rede, 4xx/5xx) por `limite_falhas` vezes seguidas, ou o teto de
    chamadas do próprio orçamento, DESLIGA o 2º revisor pelo resto da execução
    e o pipeline segue só com validador + revisor. Cada questão decidida sem
    ele sai marcada (`indisponivel`). Resposta cortada/malformada do Gemini
    NÃO é indisponibilidade (é conteúdo): continua reprovando, fail-closed."""
    arb = montar_arbitro(dry_run=dry_run, max_chamadas=max_chamadas,
                         log_uso=log_uso or LOG_USO_GEMINI.with_name("uso_api_gemini_revisor2.jsonl"),
                         cache_path=cache_path or CACHE_ARBITRO.with_name("gemini_revisor2_cache.jsonl"),
                         modelo=modelo)
    estado = {"ativo": True, "falhas": 0, "motivo": None, "sem_revisor2": 0}
    agentes.segundo_revisor_estado = estado

    def _desligar(motivo):
        estado.update(ativo=False, motivo=motivo)
        aviso(f"!!! 2º revisor (Gemini) INDISPONÍVEL: {motivo}. Seguindo SEM ele (validador + revisor).")

    def segundo(q, ano, hab, desc, sub):
        if not estado["ativo"]:
            estado["sem_revisor2"] += 1
            return {"indisponivel": True, "veredito": None, "motivo": estado["motivo"]}
        try:
            r = arb.revisar_cego(agentes, q, ano, hab, desc, sub)
        except aq.OrcamentoEsgotado as exc:
            _desligar(f"orçamento de chamadas Gemini esgotado ({exc})")
            estado["sem_revisor2"] += 1
            return {"indisponivel": True, "veredito": None, "motivo": estado["motivo"]}
        if r.get("erro") == "erro_api":
            estado["falhas"] += 1
            detalhe = (r.get("problemas") or [{}])[0].get("detalhe", "erro_api")
            if estado["falhas"] >= limite_falhas:
                _desligar(f"{estado['falhas']} falhas de API seguidas ({detalhe})")
            estado["sem_revisor2"] += 1
            return {"indisponivel": True, "veredito": None, "motivo": f"erro_api: {detalhe}"[:200]}
        estado["falhas"] = 0
        return r

    agentes.segundo_revisor = segundo
    return arb
