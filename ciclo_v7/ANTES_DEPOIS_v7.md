# Candidata v7 × v6: antes e depois

Gerado por `src/verificar_v7.py` (sem API).

## Critérios de aprovação

| critério | resultado |
|---|---|
| V7-01 schema igual ao da v6 | PASSA |
| V7-02 sem enunciado duplicado e sem contaminação com val* | PASSA |
| V7-03a REFUTADO = 0 e FRACO = 0 | **FALHA** |
| V7-03b matematicamente_problematica = 0 | PASSA |
| V7-03c % verificável >= v6 | PASSA |
| V7-03d pouco_explicativas <= v6 | PASSA |
| V7-03e near-dup pares <= v6 | PASSA |
| V7-03f letras entre 18% e 22% | PASSA |
| V7-03g explicabilidade dos distratores >= v6 | PASSA |

**Candidata NÃO APROVADA pelos critérios automáticos.**

## Antes × depois

| métrica | v6 | v7 |
|---|---|---|
| exemplos | 2807 | 2771 |
| questoes | 2849 | 2813 |
| verificavel_pct | 63.4 | 63.9 |
| near_dup_pares | 45 | 45 |
| auditoria_gabarito | {'CONFIRMADO': 1811, 'CEGO': 99, 'JUIZES': 939} | {'CONFIRMADO': 1802, 'CEGO': 95, 'REFUTADO': 2, 'JUIZES': 914} |
| resolucoes | {'adequada': 2802, 'correta_pouco_explicativa': 24, 'incompleta': 23, 'matematicamente_problematica': 0, 'nao_avaliavel': 0} | {'adequada': 2781, 'correta_pouco_explicativa': 18, 'incompleta': 14, 'matematicamente_problematica': 0, 'nao_avaliavel': 0} |
| letras | {'B': 571, 'D': 570, 'A': 570, 'C': 569, 'E': 569} | {'B': 566, 'C': 565, 'A': 562, 'E': 561, 'D': 559} |
| diversidade.n | 2849 | 2813 |
| diversidade.subtemas_distintos | 231 | 231 |
| diversidade.entropia_subtema | 0.946 | 0.945 |
| diversidade.entropia_raciocinio | 0.654 | 0.653 |
| diversidade.entropia_estrutura | 0.58 | 0.582 |
| diversidade.entropia_representacao | 0.506 | 0.502 |
| diversidade.entropia_contexto | 0.929 | 0.927 |
| diversidade.valores_numericos_distintos | 0.638 | 0.64 |
| diversidade.moldes_distintos | 0.993 | 0.993 |
| diversidade.pct_outros | 19.5 | 19.6 |
| distratores.explicabilidade_pct | 60.0 | 60.1 |
| distratores.nda | 24.3 | 24.7 |
| distratores.revisao | 0.0 | 0.0 |
