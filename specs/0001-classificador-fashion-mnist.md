# 0001 — Classificador Fashion-MNIST: treinar, exportar, verificar

- **Estado:** **rascunho — aguardando aprovação** (uma decisão sua no fim)
- **Data:** 2026-09-16

> **Primeira vez com ML?** Leia [`docs/CONCEITOS.md`](../docs/CONCEITOS.md) antes desta
> spec. Ela usa uns quinze termos — *baseline*, quantização, *split*, *recall* — que o
> mapa de conceitos traduz para o que você já conhece. A spec não fica mais fácil, mas
> deixa de ser um muro.

## Problema

O repositório não tem nada. E o objetivo do repositório — entender o que acontece entre
um modelo treinado e um número na tela do celular — não se aprende pelo meio: precisa de
**uma fatia vertical completa**, pequena o suficiente para caber num dia e real o
suficiente para doer nos lugares certos.

Os lugares que doem, e que esta etapa existe para expor:

1. O modelo treinado em Python e o modelo que roda no celular **não são o mesmo modelo**.
   A quantização muda os números. Quanto, e onde isso é aceitável, é uma decisão — não
   um detalhe.
2. Pré-processamento em dois lugares é a causa nº1 de "funcionava no notebook".
3. Uma acurácia sem baseline não é um resultado, é um número.

## Objetivo

Treinar um classificador de peças de roupa, exportá-lo como `.tflite` quantizado, e
**provar por teste** que o modelo exportado concorda com o original dentro de um limite
declarado antes de treinar.

## Fora de âmbito

- **Qualquer código Android.** Esta etapa termina em `models/`. O app é a `0002`.
- Câmera, galeria, imagens do mundo real. A entrada aqui é o dataset.
- Treino on-device, *transfer learning*, aumento de dados, ajuste de hiperparâmetro.
- GPU/NNAPI *delegates*, medição de latência — entram na `0002`, que é onde há hardware.
- Qualquer tentativa de maximizar acurácia. A meta é um **piso**, e bater o piso encerra
  o assunto.

## Dados

- **Origem:** `tf.keras.datasets.fashion_mnist` — 10 classes de peças de roupa,
  imagens 28×28 em tons de cinza, `uint8` em `[0, 255]`.
- **Tamanho:** 60 000 de treino, 10 000 de teste, classes balanceadas (6 000 / 1 000 por
  classe).
- **Splits:** o dataset já vem com treino e teste separados. A validação sai das
  **últimas 6 000 imagens do treino** — nunca do teste.
  - treino efetivo: 54 000 · validação: 6 000 · teste: 10 000
  - O conjunto de teste é tocado **uma única vez**, no fim, para reportar o número. Não
    se escolhe nada olhando para ele (Regra 01, item 1).
- **Labels:** em português, na ordem do dataset — `camiseta`, `calça`, `pulôver`,
  `vestido`, `casaco`, `sandália`, `camisa`, `tênis`, `bolsa`, `bota`.

> Por que Fashion-MNIST e não MNIST: os dígitos do MNIST são fáceis demais — qualquer
> coisa acerta 98% e nenhuma escolha sua muda o resultado, o que torna o exercício mudo.
> Fashion-MNIST tem o mesmo formato (logo o mesmo custo de infra) e erra o suficiente
> para as decisões aparecerem nos números. E confunde `camiseta`/`camisa`/`pulôver`, o
> que faz a matriz de confusão ensinar algo.

## Modelo e números

CNN pequena: duas camadas convolucionais (32 e 64 filtros, 3×3) com *max pooling*,
`Dense(128)`, `Dropout(0.3)`, `Dense(10)` com *softmax*. ~10 épocas, `Adam`,
`sparse_categorical_crossentropy`. Seed fixa em `42`.

| | Valor | Origem |
|---|---|---|
| Baseline — classe majoritária | **0,10** | classes balanceadas, 10 classes |
| Baseline — regressão logística | **≈ 0,84** | referência conhecida do dataset; será medida, não assumida |
| **Meta (piso) do modelo Keras no teste** | **≥ 0,88** | acima do baseline linear por margem que não é ruído |
| **Perda aceitável na quantização int8** | **≤ 1 ponto percentual** de acurácia | ver Decisões |
| **Concordância de classe predita Keras ↔ tflite** | **≥ 98%** das 10 000 imagens de teste | ver Decisões |

A regressão logística é medida de propósito: é ela que responde "a CNN valeu a pena?".
Se a CNN ficar a 1 ponto do modelo linear, a resposta é não — e isso é um resultado
legítimo desta etapa, não um fracasso.

## Contrato do artefato

Vai para `models/`, no mesmo commit do script que o gerou (Regra 01):

- `fashion_mnist_cnn_int8.tflite`
- `fashion_mnist_cnn_int8.json` com: versão, shape/dtype de entrada e saída, escala e
  *zero point* da quantização, as 10 labels, acurácia no teste, acurácia do baseline,
  e o SHA do commit de treino.

**Entrada:** `[1, 28, 28, 1]`, `uint8` em `[0, 255]`.
A divisão por 255 é uma camada `Rescaling` **dentro do grafo** — o lado Android entrega
o pixel cru e não replica aritmética nenhuma (Regra 01).

## Decisões

| Decisão | Alternativa posta de lado | Motivo |
|---|---|---|
| Quantização **int8 completa** (pesos e ativações), com dataset representativo | Só `float16`, ou nenhuma quantização | float16 quase não perde precisão — e por isso não ensina nada sobre o trade-off. int8 é o que se usa de verdade em celular (4× menor, e é o que os *delegates* aceleram) e é o que **força** o teste de invariância a existir |
| A normalização mora no grafo (`Rescaling`) | Normalizar no Python antes do treino e repetir no Kotlin | Duas implementações da mesma multiplicação divergem, e divergem em silêncio. No grafo, é impossível o treino e a inferência discordarem (Regra 01) |
| Limite de invariância declarado **antes** de treinar (1pp / 98%) | Treinar, medir a perda, e chamar o resultado de limite | Limite escolhido depois do número é o número com outro nome. Se o limite for estourado, a saída é investigar ou renegociar o limite **explicitamente** — não ajustá-lo em silêncio |
| O conjunto de teste é usado uma vez só | Acompanhar a acurácia de teste a cada época | Olhar para o teste e decidir qualquer coisa a partir dele é vazamento pelo operador. É para isso que existe a validação |
| Métrica reportada como **piso** (`≥ 0,88`), nunca igualdade | `assert acuracia == 0.9137` | Treino tem ruído mesmo com seed; igualdade dá teste frágil que quebra sem nada estar errado (Regra 02) |
| Matriz de confusão faz parte da entrega | Só a acurácia | 0,90 pode ser um modelo bom ou um modelo que ignora uma classe inteira. A acurácia não distingue os dois casos; a matriz distingue |
| Seed fixa e registrada | Deixar aleatório | Sem seed, não se sabe se a melhora foi sua ou sorte |

## Critérios de aceite

- [ ] **CA-01** — Dado o dataset carregado, quando os splits são construídos, então há
      54 000 / 6 000 / 10 000 exemplos e **nenhum índice do teste aparece em treino ou
      validação**.
- [ ] **CA-02** — Dado o baseline de classe majoritária, quando é avaliado no teste,
      então sua acurácia é reportada e fica em ≈ 0,10.
- [ ] **CA-03** — Dado o baseline de regressão logística, quando é treinado e avaliado,
      então sua acurácia é reportada — e é esse o número que o modelo precisa superar.
- [ ] **CA-04** — Dado o modelo Keras treinado, quando é avaliado **uma vez** no teste,
      então a acurácia é **≥ 0,88** e supera o baseline logístico.
- [ ] **CA-05** — Dado o modelo treinado, quando é exportado, então existe um
      `.tflite` int8 cuja entrada é `[1,28,28,1] uint8` e a saída `[1,10]`, exatamente
      como declarado no JSON de metadata.
- [ ] **CA-06** — Dado o mesmo conjunto de teste, quando é passado pelo Keras e pelo
      `.tflite`, então as classes preditas concordam em **≥ 98%** dos casos e a acurácia
      do `.tflite` não fica mais de **1 ponto percentual** abaixo da do Keras.
- [ ] **CA-07** — Dado que a normalização está no grafo, quando se passa um pixel cru
      `uint8` ao `.tflite`, então a predição é correta **sem** qualquer divisão por 255
      feita fora do modelo.
- [ ] **CA-08** — Dado o `.tflite` exportado, quando o JSON ao lado é lido, então ele
      contém labels, shapes, dtypes, acurácia, baseline e o SHA do commit de treino.
- [ ] **CA-09** — Dado o mesmo comando de treino executado duas vezes com a mesma seed,
      quando as acurácias são comparadas, então diferem em menos de 0,5 ponto percentual.
- [ ] **CA-10** — Dado o modelo avaliado, quando a matriz de confusão é gerada, então
      nenhuma classe tem *recall* igual a zero.

## Plano de testes

| Alvo | O que verificar |
|---|---|
| `dados.py` | Shapes, dtype `uint8`, valores em `[0,255]`, tamanhos dos splits, e a **não-intersecção** treino/teste (CA-01) |
| `baseline.py` | Classe majoritária dá ≈ 0,10 (CA-02); logística reporta um número (CA-03) |
| `export.py` | Shape e dtype de entrada/saída batem com o JSON (CA-05, CA-08) |
| `verify.py` | **O teste central:** concordância ≥ 98% e queda ≤ 1pp (CA-06); pixel cru funciona sem normalização externa (CA-07) |
| Treino | Piso de acurácia (CA-04); reprodutibilidade com seed (CA-09); nenhuma classe com recall zero (CA-10) |

Verificação obrigatória antes de chamar a etapa de pronta: **quebrar de propósito** a
camada `Rescaling` (removê-la do grafo) e confirmar que o CA-07 fica vermelho. Se ficar
verde, o teste não estava olhando para o que diz olhar.

## Riscos e questões em aberto

- **O Python é o 3.9.6 do sistema.** É o único arm64 nesta máquina (ver `CLAUDE.md`).
  Funciona — TensorFlow 2.20 instalado e verificado — mas é uma versão em fim de vida.
  *Plano de saída em `training/README.md`.*
- **Sem GPU.** Treino em CPU, ~2–4 min para esta CNN. Aceitável nesta escala; deixa de
  ser na primeira vez que o modelo crescer.
- **A quantização int8 pode não bater o limite de 1pp.** É o risco real desta etapa — e
  se acontecer, é a parte mais instrutiva dela: a saída não é afrouxar o limite, é
  entender se o dataset representativo era pequeno demais ou se o modelo é sensível.

### Decisões que eram suas — e por que eu as resolvi

Escrevi a primeira versão desta spec com três pontos em aberto para você decidir. Não
faz sentido: as três exigem intuição de quem já treinou modelo antes, e você disse que
está começando. Pedir que decida isso agora seria pedir um chute e chamá-lo de decisão.

Então resolvi, com o motivo — e é isso que você aprova ou contesta:

| Ponto | Resolução | Por quê |
|---|---|---|
| A meta é 0,88 ou 0,90? | **0,88** | O objetivo da etapa é a fronteira Python↔Android, não caçar acurácia. 0,88 é folgado para esta CNN (~0,91 é o normal): você bate no primeiro treino e segue. Se virar fácil demais, a etapa 4 tem onde apertar |
| Cortar o CA-09 (reprodutibilidade) e o CA-10 (recall por classe)? | **Ficam os dois** | São justamente os que ensinam o que ML tem de diferente: o CA-09 mostra que treino tem ruído, e o CA-10 mostra que acurácia esconde uma classe inteira ignorada. Cortá-los deixava a spec mais curta e mais pobre |
| Vale medir a regressão logística? | **Vale** | É o número que responde "a rede neural valeu a pena?". Sem ele, `0,91` não tem escala: pode ser excelente ou pode ser pior que um modelo de 20 linhas. Custa ~10 linhas de código |

### A única decisão que ainda é sua

**Quanto você quer ver acontecer antes de entender o que aconteceu?**

Duas formas de rodar a etapa 1, e a escolha é de estilo de aprendizado, não técnica:

- **(a) Passo a passo.** Implementamos um script por vez — carregar dados, ver os shapes,
  rodar o baseline burro, treinar, exportar, comparar. Você roda cada um e vê o número
  antes de seguir. Mais lento, e você entende cada pedaço quando ele aparece.
- **(b) Pipeline inteiro primeiro.** Implemento os seis passos, você roda uma vez, vê o
  resultado final, e *depois* voltamos a abrir cada parte. Mais rápido para ter a coisa
  funcionando, e o entendimento vem na segunda passada.

Não há resposta certa. Para quem nunca viu um modelo treinar, **(a)** costuma render
mais — o "aha" do baseline burro acertando 10% e do modelo pulando para 91% só funciona
se você vir os dois números separados, em momentos separados.

## Riscos e questões em aberto (continuação)

- **A spec é densa para uma primeira etapa.** É de propósito: o `docs/CONCEITOS.md`
  existe para absorver essa densidade. Se ainda assim travar, o problema é a spec, não
  você — e a saída é me dizer onde travou.
