# NNNN — <nome da etapa>

- **Estado:** rascunho | aprovada | implementada
- **Data:**

## Problema
O que não dá para fazer hoje, e por que isso importa.

## Objetivo
Uma frase, em comportamento observável.

## Fora de âmbito
O que esta spec explicitamente **não** resolve.

## Dados
Origem, tamanho, splits (e como se garante que não se intersectam), formato, dtype,
intervalo de valores.

## Modelo e números
| | Valor | Origem |
|---|---|---|
| Baseline burro | | classe majoritária / modelo linear |
| Meta (piso) | | |
| Perda aceitável na quantização | | |

Sem baseline, a meta não significa nada (Regra 02).

## Contrato do artefato
O que vai para `models/`: nome, shape/dtype de entrada e saída, pré-processamento
esperado, labels. É o conteúdo do JSON de metadata (Regra 01).

## Decisões
| Decisão | Alternativa posta de lado | Motivo |
|---|---|---|

## Critérios de aceite
- [ ] **CA-01** — Dado …, quando …, então … (observável, verificável por teste)

## Plano de testes
| Alvo | O que verificar |
|---|---|

## Riscos e questões em aberto
Inclusive o que ficou **por decidir** — e de quem é a decisão.
