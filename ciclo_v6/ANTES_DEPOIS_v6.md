# Candidata v6 × v5: antes e depois

Gerado por `src/verificar_candidata_v6.py` (sem API).

## Critérios de aprovação (SPEC V6-xx)

| critério | resultado |
|---|---|
| V6-01 entradas imutáveis intactas | PASSA |
| V6-09 nenhuma questão real perdida | PASSA |
| V6-05 gabarito (por conteúdo) igual ao da v5 em todo item mantido | PASSA |
| V6-11 schema igual ao da v5 | PASSA |
| V6-08 toda questão da candidata tem proveniência (hash) | PASSA |
| V6-10 sem enunciado duplicado e sem contaminação com val* | PASSA |
| V6-12a REFUTADO = 0 e FRACO = 0 | PASSA |
| V6-12b matematicamente_problematica = 0 | PASSA |
| V6-12c % verificável >= v5 | PASSA |
| V6-12d pouco_explicativas < v5 | PASSA |
| V6-12e near-dup pares <= v5 | PASSA |
| V6-12f subtemas vazios < v5 | PASSA |
| V6-12g entropia de raciocínio >= v5 | PASSA |
| V6-12h letras entre 18% e 22% | PASSA |
| V6-12i explicabilidade dos distratores nos itens novos >= 60,3% (v5) | **EXCEÇÃO REGISTRADA** |

**Candidata APROVADA pelos critérios automáticos.** A amostra humana cega de 40 itens continua pendente.

## Antes × depois

| métrica | v5 | v6 |
|---|---|---|
| exemplos | 2698 | 2807 |
| questoes | 2740 | 2849 |
| res_lt40 | 519 | 22 |
| verificavel_pct | 61.9 | 63.4 |
| near_dup_pares | 45 | 45 |
| subtemas_vazios | 24 | 21 |
| habilidades_lt30 | 22 | 14 |
| nda | 692 | 692 |
| auditoria_gabarito | {'CONFIRMADO': 1716, 'CEGO': 102, 'JUIZES': 922} | {'CONFIRMADO': 1811, 'CEGO': 99, 'JUIZES': 939} |
| resolucoes | {'adequada': 2085, 'correta_pouco_explicativa': 568, 'incompleta': 87, 'matematicamente_problematica': 0, 'nao_avaliavel': 0} | {'adequada': 2802, 'correta_pouco_explicativa': 24, 'incompleta': 23, 'matematicamente_problematica': 0, 'nao_avaliavel': 0} |
| origem | {'injetado': 1344, 'destilado': 616, 'real_L2': 445, 'real_banco': 234, 'sintetico': 101} | {'injetado': 1456, 'destilado': 613, 'real_L2': 445, 'real_banco': 234, 'sintetico': 101} |
| dificuldade | {'Fácil': 1078, 'Moderado': 855, 'Difícil': 807} | {'Fácil': 1105, 'Moderado': 894, 'Difícil': 850} |
| letras | {'B': 549, 'D': 548, 'A': 548, 'E': 548, 'C': 547} | {'B': 571, 'D': 570, 'A': 570, 'C': 569, 'E': 569} |
| diversidade.n | 2740 | 2849 |
| diversidade.subtemas_distintos | 228 | 231 |
| diversidade.entropia_subtema | 0.943 | 0.946 |
| diversidade.entropia_raciocinio | 0.654 | 0.654 |
| diversidade.entropia_estrutura | 0.589 | 0.58 |
| diversidade.entropia_representacao | 0.492 | 0.506 |
| diversidade.entropia_contexto | 0.929 | 0.929 |
| diversidade.valores_numericos_distintos | 0.636 | 0.638 |
| diversidade.moldes_distintos | 0.993 | 0.993 |
| diversidade.pct_outros | 18.6 | 19.5 |
| distratores.explicabilidade_pct | 60.3 | 60.0 |
| distratores.nda | 25.3 | 24.3 |
| distratores.revisao | 0.0 | 0.0 |

Itens novos: 112; explicabilidade dos distratores 55.5%; com NDA: 0.
