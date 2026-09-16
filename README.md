# AndroidMlLab

Aprender **machine learning de ponta a ponta, com Android como destino**.

```
dados  →  treino (Python/Keras)  →  export (.tflite int8)  →  inferência on-device (Kotlin)
```

A escolha por trás do repositório: não começar pelo ML Kit. As APIs prontas resolvem o
problema sem ensinar nada sobre ele — você chama um método e recebe uma resposta, e o
modelo continua sendo uma caixa preta. Aqui o modelo é seu: você treina, mede, quantiza,
e descobre no caminho que o modelo que roda no celular **não é** o que você treinou.

> **Nunca treinou um modelo?** É o caso previsto. Leia primeiro
> [`docs/VISAO-GERAL.md`](docs/VISAO-GERAL.md) (o problema e o plano, sem jargão), depois
> [`docs/CONCEITOS.md`](docs/CONCEITOS.md): é o mapa dos termos que aparecem nas specs
> — *baseline*, quantização, *split*, tensor — traduzidos para o que você já conhece.
> Leva ~10 min e é o que faz o resto deixar de ser um muro.

## O roteiro, em quatro etapas

| Etapa | O que se constrói | O que se aprende de verdade |
|---|---|---|
| **1. Treinar e exportar** (`specs/0001`) | CNN pequena no Fashion-MNIST → `.tflite` int8 + metadata | Splits sem vazamento, baseline, e o custo real da quantização |
| **2. Servir no Android** (`specs/0002`) | App Compose que classifica imagens empacotadas | A fronteira: pré-processamento, `uint8` vs `float`, latência medida |
| **3. Entrada do mundo real** (`0003`) | Câmera / galeria como entrada | Onde o modelo de laboratório quebra: resize, rotação, iluminação, domínio diferente |
| **4. Um modelo que vale a pena** (`0004`) | Transfer learning sobre MobileNet, dataset próprio | O ciclo completo de quem faz isso a sério — e o trade-off tamanho × precisão × latência |

Cada etapa só começa quando a anterior está verde. **A etapa 1 é a única aberta agora.**

## Por onde começar, concretamente

```bash
# 1. o que estamos fazendo e por quê, sem jargão  (~5 min)
docs/VISAO-GERAL.md

# 2. o vocabulário  (~10 min, e evita ler as specs no escuro)
docs/CONCEITOS.md

# 3. o método e as regras
CLAUDE.md
.claude/rules/01-fronteira-modelo.md
.claude/rules/02-testes.md

# 4. a spec da etapa 1
specs/0001-classificador-fashion-mnist.md

# 5. confirme que o ambiente está de pé (já instalado e verificado)
training/.venv/bin/python -m pytest training/tests -q
```

No fim da spec 0001 há **uma** decisão sua: implementar a etapa passo a passo (você roda
e vê cada número aparecer) ou o pipeline inteiro de uma vez. As outras três que eu tinha
deixado em aberto estão resolvidas lá, com o motivo — pedir que você as decidisse antes
de ter visto um modelo treinar seria pedir um chute.

A regra deste repositório: **o agente escreve,
você decide.** Só que "decidir" não pode significar escolher entre duas opções que você
ainda não tem como comparar — nesses casos eu decido e mostro o porquê, e você contesta.

## Estrutura

```
CLAUDE.md              contexto que o agente lê no início de cada conversa
.claude/rules/         duas regras: fronteira do modelo, e testes
specs/                 uma spec por etapa do pipeline
training/              Python: dados, treino, export, verificação
models/                artefatos exportados (.tflite + JSON de metadata)
app/                   Android — ainda não existe (etapa 2)
docs/VISAO-GERAL.md    o problema e o plano, sem jargão — comece por aqui
docs/CONCEITOS.md      o mapa dos termos de ML — comece por aqui
docs/RESULTADOS.md     os números de cada passo, e o que eles querem dizer
docs/ATRITO.md         onde o método e a IA falharam, e o que virou regra
```

## Ambiente

⚠️ Esta máquina é **arm64**, mas o Homebrew está em `/usr/local` (prefixo Intel): o
`python3` do PATH é x86_64 sob Rosetta e **não tem wheels de TensorFlow**. O venv deste
repo usa o `/usr/bin/python3` (arm64, 3.9.6), onde o TensorFlow 2.20 instala e roda
nativo — já verificado. Detalhes e plano de saída em [`training/README.md`](training/README.md).

## Comandos

```bash
training/.venv/bin/python -m pytest training/tests    # testes
training/.venv/bin/python -m training.src.treino      # treina  (etapa 1, após aprovar a spec)
training/.venv/bin/python -m training.src.export      # exporta .tflite
training/.venv/bin/python -m training.src.verify      # Keras vs .tflite  ← o teste que importa
```

## A ideia central

> **O agente escreve, você decide, os testes provam.**

Com uma diferença que é específica de ML, e que está na Regra 02:

> Em app comum, teste verde e comportamento errado é raro.
> Em ML é o caso normal — o erro está nos números, não no fluxo.
