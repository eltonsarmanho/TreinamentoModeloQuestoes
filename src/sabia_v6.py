"""Chamada medida à API Sabiá para os papéis NOVOS do ciclo v6 (resolução, distratores).

Não é um cliente novo: usa o cliente, o orçamento de chamadas, a sanitização de segredos e a
política de retentativa de agentes_questoes (ClienteMaritacaMedido, Orcamento, _sanitizar,
_retentavel, _ler_usage, abrir_para_gravar). agentes_questoes.Agentes só aceita os quatro
papéis da injeção (gerador/validador/validador_fase2/revisor) com max_tokens/temperatura fixos
por papel; os papéis daqui precisam de prompt versionado próprio e de um log mais completo.

O que este módulo acrescenta:
  - teto DURO em R$: antes de cada envio reserva o PIOR caso (entrada estimada + max_tokens de
    saída ao preço do modelo); se passar do teto, nada é enviado (OrcamentoEsgotado);
  - log por chamada com: modelo, papel, versão do prompt, hash do exemplo, codigo_item,
    tokens de entrada/saída, latência, custo estimado, finish_reason, e depois resultado e
    motivo (registrar_resultado);
  - fail-closed: erro de API, timeout, JSON inválido, saída truncada ou teto atingido devolvem
    None — quem chama trata None como "sem decisão" (abstenção), nunca como aprovação;
  - cache append-only por (papel, modelo, versão do prompt, mensagens, max_tokens, temperatura):
    só resposta completa e interpretável entra; retomar um lote não paga de novo;
  - dry-run: um respondedor local por papel, sem rede (custo 0, log marcado simulado).

Preços: ciclo_v6/precos_maritaca.json (nada fixo no código).
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRECOS_PADRAO = ROOT / "ciclo_v6" / "precos_maritaca.json"
CHARS_POR_TOKEN = 3.6  # mesma estimativa de agentes_questoes._estimar_tokens


class OrcamentoEsgotado(RuntimeError):
    """O próximo envio passaria do teto de chamadas ou de R$: nada foi enviado."""


def versao_prompt(*textos):
    return hashlib.sha1("\x1e".join(textos).encode("utf-8")).hexdigest()[:12]


def carregar_precos(caminho=PRECOS_PADRAO):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))["modelos"]


def custo_brl(precos, modelo, prompt_tokens, completion_tokens):
    p = precos.get(modelo)
    if p is None:
        raise KeyError(f"modelo sem preço em {PRECOS_PADRAO.name}: {modelo}")
    return (prompt_tokens * p["entrada"] + completion_tokens * p["saida"]) / 1e6


def estimar_tokens(texto):
    return int(len(texto or "") / CHARS_POR_TOKEN) + 1


class ChamadorSabia:
    """papeis: {papel: {"modelo", "max_tokens", "temperatura", "versao_prompt"}}.
    clientes: {papel: cliente com chat_completion(messages, max_tokens, temperature)}.
    simulado: True = clientes locais (dry-run); o log marca e o custo é 0."""

    def __init__(self, papeis, clientes, max_chamadas, teto_brl, precos, log_uso=None, cache=None,
                 tentativas=3, backoff=2.0, dormir=time.sleep, simulado=False):
        if max_chamadas is None or int(max_chamadas) <= 0:
            raise ValueError("max_chamadas > 0 é obrigatório (teto duro)")
        if not simulado and (teto_brl is None or float(teto_brl) <= 0):
            raise ValueError("modo real exige teto_brl > 0")
        self.papeis, self.clientes, self.precos = papeis, clientes, precos
        self.max_chamadas, self.teto_brl = int(max_chamadas), (None if teto_brl is None else float(teto_brl))
        self.chamadas, self.gasto_brl, self.reservado_brl = 0, 0.0, 0.0
        self.log_uso = Path(log_uso) if log_uso else None
        self.cache_path = Path(cache) if cache else None
        self.tentativas, self.backoff, self.dormir, self.simulado = max(1, int(tentativas)), backoff, dormir, simulado
        self.cache = {}
        self.abstencoes = {}
        if self.cache_path and self.cache_path.exists():
            for linha in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue  # linha final cortada por queda: ignorada
                if r.get("chave") and r.get("texto") is not None:
                    self.cache[r["chave"]] = r
        for papel, cfg in papeis.items():
            if not simulado:
                custo_brl(precos, cfg["modelo"], 0, 0)  # falha cedo se o modelo não tem preço

    # ------------------------------------------------------------------ auxiliares
    def _chave(self, papel, mensagens):
        cfg = self.papeis[papel]
        bruto = json.dumps([papel, cfg["modelo"], cfg["versao_prompt"], cfg["max_tokens"], cfg["temperatura"], mensagens],
                           ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(bruto.encode("utf-8")).hexdigest()

    def pior_caso_brl(self, papel, mensagens):
        if self.simulado:
            return 0.0
        cfg = self.papeis[papel]
        entrada = sum(estimar_tokens(m.get("content", "")) for m in mensagens)
        return custo_brl(self.precos, cfg["modelo"], entrada, cfg["max_tokens"])

    def _reservar(self, papel, mensagens):
        if self.chamadas + 1 > self.max_chamadas:
            raise OrcamentoEsgotado(f"teto de {self.max_chamadas} chamadas atingido")
        pior = self.pior_caso_brl(papel, mensagens)
        if self.teto_brl is not None and self.gasto_brl + pior > self.teto_brl:
            raise OrcamentoEsgotado(f"teto de R$ {self.teto_brl:.2f} seria ultrapassado (gasto {self.gasto_brl:.4f} "
                                    f"+ pior caso {pior:.4f})")
        self.chamadas += 1
        return pior

    def _log(self, reg):
        if not self.log_uso:
            return
        import agentes_questoes as aq
        with aq.abrir_para_gravar(self.log_uso, "a") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ chamada
    def chamar(self, papel, mensagens, *, hash_exemplo, codigo_item=None, interpretavel=lambda t: True):
        """Texto da resposta, ou None (abstenção: erro, truncada, ininterpretável, teto).
        OrcamentoEsgotado é levantado ANTES de enviar e deve interromper o lote."""
        import agentes_questoes as aq
        cfg = self.papeis[papel]
        chave = self._chave(papel, mensagens)
        if chave in self.cache:
            return self.cache[chave]["texto"]
        for tentativa in range(1, self.tentativas + 1):
            self._reservar(papel, mensagens)
            t0 = time.monotonic()
            reg = {"ts": aq.agora_iso(), "papel": papel, "modelo": cfg["modelo"], "versao_prompt": cfg["versao_prompt"],
                   "hash_exemplo": hash_exemplo, "codigo_item": codigo_item, "tentativa": tentativa,
                   "simulado": self.simulado}
            try:
                resp = self.clientes[papel].chat_completion(messages=mensagens, max_tokens=cfg["max_tokens"],
                                                            temperature=cfg["temperatura"])
                texto = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 — rede/cota/provedor; a chave nunca é impressa
                erro = aq._sanitizar(f"{type(exc).__name__}: {exc}")
                # sem usage: cobra o pior caso (o provedor pode ter cobrado)
                custo = self.pior_caso_brl(papel, mensagens)
                self.gasto_brl += custo
                reg.update(ok=False, erro=erro, latencia_s=round(time.monotonic() - t0, 2), prompt_tokens=0,
                           completion_tokens=0, estimado=True, custo_brl=round(custo, 6))
                self._log(reg)
                if tentativa < self.tentativas and aq._retentavel(exc):
                    self.dormir(self.backoff * (2 ** (tentativa - 1)))
                    continue
                self.abstencoes[hash_exemplo] = "erro_api"
                return None
            usage = aq._ler_usage(resp)
            estimado = usage is None
            if estimado:
                usage = (sum(estimar_tokens(m.get("content", "")) for m in mensagens), estimar_tokens(texto))
            custo = 0.0 if self.simulado else custo_brl(self.precos, cfg["modelo"], usage[0], usage[1])
            self.gasto_brl += custo
            fim = getattr(resp.choices[0], "finish_reason", None)
            truncada = aq.resposta_truncada({"finish_reason": fim})
            ok = not truncada and interpretavel(texto)
            reg.update(ok=ok, latencia_s=round(time.monotonic() - t0, 2), prompt_tokens=usage[0],
                       completion_tokens=usage[1], estimado=estimado, custo_brl=round(custo, 6),
                       finish_reason=None if fim is None else str(fim))
            if not ok:
                reg["erro"] = "resposta_truncada" if truncada else "resposta_ininterpretavel"
            self._log(reg)
            if not ok:
                self.abstencoes[hash_exemplo] = reg["erro"]
                return None
            if self.cache_path:
                r = {"chave": chave, "papel": papel, "modelo": cfg["modelo"], "versao_prompt": cfg["versao_prompt"],
                     "texto": texto, "prompt_tokens": usage[0], "completion_tokens": usage[1], "ts": reg["ts"]}
                self.cache[chave] = r
                with aq.abrir_para_gravar(self.cache_path, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return texto
        return None

    def registrar_resultado(self, papel, hash_exemplo, codigo_item, resultado, motivo):
        """Linha de desfecho no mesmo log: o que a resposta decidiu e por quê."""
        import agentes_questoes as aq
        self._log({"ts": aq.agora_iso(), "papel": papel, "modelo": self.papeis[papel]["modelo"],
                   "versao_prompt": self.papeis[papel]["versao_prompt"], "hash_exemplo": hash_exemplo,
                   "codigo_item": codigo_item, "tipo": "resultado", "resultado": resultado, "motivo": motivo,
                   "simulado": self.simulado})

    def resumo(self):
        return {"chamadas": self.chamadas, "max_chamadas": self.max_chamadas, "gasto_brl": round(self.gasto_brl, 4),
                "teto_brl": self.teto_brl, "simulado": self.simulado, "abstencoes": len(self.abstencoes)}


def estimar_lote(n_itens, papeis, precos, tokens_medidos, retentativas=1.15):
    """Custo previsto (R$) e chamadas para n_itens, por papel.
    tokens_medidos: {papel: (entrada_mediana, saida_mediana, saida_p90)} — medidos nos logs reais."""
    out, total, total_p90, chamadas = {}, 0.0, 0.0, 0
    for papel, cfg in papeis.items():
        ent, sai, sai90 = tokens_medidos[papel]
        n = int(round(n_itens * cfg.get("chamadas_por_item", 1) * retentativas))
        c = n * custo_brl(precos, cfg["modelo"], ent, sai)
        c90 = n * custo_brl(precos, cfg["modelo"], ent, sai90)
        out[papel] = {"modelo": cfg["modelo"], "chamadas": n, "custo_mediano_brl": round(c, 2), "custo_p90_brl": round(c90, 2)}
        total, total_p90, chamadas = total + c, total_p90 + c90, chamadas + n
    return {"por_papel": out, "chamadas": chamadas, "custo_mediano_brl": round(total, 2), "custo_p90_brl": round(total_p90, 2)}
