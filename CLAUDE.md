# AndroidMlLab

Laboratório de aprendizado: **machine learning de ponta a ponta com Android como destino**.
Não é um app de produção — é o repositório onde se aprende o ciclo completo:

```
dados  →  treino (Python/Keras)  →  export (.tflite quantizado)  →  inferência on-device (Kotlin)
```

O objetivo não é entregar telas. É entender o que acontece entre um `.h5` e um número na
tela do celular — e saber provar que continua correto depois de cada etapa.

## Como trabalhamos aqui

Spec-first, mas enxuto: **duas regras** e uma spec por
etapa do pipeline.

```
ideia  →  spec em specs/NNNN-nome.md  →  revisão da spec  →  implementação  →  verde
```

- **Nenhuma etapa do pipeline começa por código.** A spec vem primeiro, mesmo quando
  parece óbvio — em ML o "óbvio" é onde mora o vazamento de dados.
- **A spec é a fonte da verdade.** Divergiu do código? Um dos dois muda, explicitamente.
- Em ML há uma armadilha que não existe em app comum: **código verde e modelo errado**.
  Por isso toda spec aqui tem, além dos critérios de aceite, um **número** que precisa ser
  batido e um **teste de invariância** entre o modelo Keras e o `.tflite` exportado.

## Quem trabalha aqui

A pessoa que mantém este repositório é desenvolvedora Android experiente e está
**começando em machine learning agora**. Isso muda como você deve responder:

- Termo de ML novo na conversa → explique em uma linha, ou aponte para
  [`docs/CONCEITOS.md`](docs/CONCEITOS.md). Não presuma *baseline*, *overfitting*,
  quantização, *recall*.
- **Não peça decisões que exigem experiência que ela ainda não tem.** Escolher um piso de
  acurácia ou um limite de quantização, sem nunca ter treinado nada, é um chute disfarçado
  de decisão. Nesses casos: **decida, mostre o motivo, e deixe-a contestar.**
- As decisões que continuam sendo dela são as de *produto* e de *estilo de aprendizado* —
  o que o app faz, quanto quer ver antes de entender, onde quer ir mais devagar.
- Analogias com desenvolvimento Android/Kotlin funcionam melhor do que analogias com
  matemática. Use-as.

## Regras obrigatórias

| Arquivo | Cobre |
|---|---|
| [.claude/rules/01-fronteira-modelo.md](.claude/rules/01-fronteira-modelo.md) | Onde mora o modelo, o que atravessa a fronteira Python↔Android, pré-processamento |
| [.claude/rules/02-testes.md](.claude/rules/02-testes.md) | Given/When/Then, o que testar em ML, o que é teste inútil aqui |

## Estrutura

```
training/       Python: dados, treino, export, verificação
├── .venv/      ambiente arm64 (Python 3.9 do sistema — ver README)
├── src/        código de treino e export
└── tests/      pytest
models/         artefatos exportados e versionados (.tflite + metadata)
app/            Android: consome o .tflite (ainda não existe — spec 0002)
specs/          uma spec por etapa
docs/CONCEITOS.md  mapa dos termos de ML para quem está começando
docs/ATRITO.md     registro de atrito: onde a IA/o método falharam
```

## Ambiente

⚠️ **Esta máquina é arm64, mas o Homebrew está em `/usr/local` (prefixo Intel).**
O `python3` do PATH é x86_64 sob Rosetta e **não** tem wheels de TensorFlow. O venv deste
repo usa o único Python nativo disponível:

```bash
/usr/bin/python3 -m venv training/.venv    # arm64, 3.9.6
```

Ver `training/README.md` para o plano de saída dessa limitação.

## Comandos

```bash
training/.venv/bin/python -m pytest training/tests    # testes do lado Python
training/.venv/bin/python -m training.src.train       # treina
training/.venv/bin/python -m training.src.export      # exporta .tflite
training/.venv/bin/python -m training.src.verify      # Keras vs .tflite
```

## Estado atual

- 📝 `specs/0001-classificador-fashion-mnist.md` — **escrita, aguardando aprovação**
- ⬜ Etapa Android — será a `0002`, depois da 0001 verde
