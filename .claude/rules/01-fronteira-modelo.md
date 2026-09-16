# Regra 01 — A fronteira do modelo

## As duas metades e o contrato entre elas

```
training/  (Python)          models/            app/  (Kotlin)
treina, avalia, exporta  →  .tflite + JSON  →  carrega e infere
```

O único canal entre as metades é a pasta `models/`. Nada mais atravessa: nem script
Python chamado do Android, nem lógica de negócio duplicada nos dois lados.

## O artefato exportado nunca vai sozinho

Todo `.tflite` em `models/` é acompanhado de um `<nome>.json` com, no mínimo:

```json
{
  "modelo": "fashion_mnist_cnn",
  "versao": 1,
  "input":  { "shape": [1, 28, 28, 1], "dtype": "uint8", "normalizacao": "x/255.0 aplicada no grafo" },
  "output": { "shape": [1, 10], "dtype": "uint8", "escala": 0.00390625, "zero_point": 0 },
  "labels": ["camiseta", "calca", "..."],
  "acuracia_teste": 0.0,
  "commit_treino": "sha"
}
```

**Motivo:** um `.tflite` não diz que pré-processamento espera. Sem esse JSON, o lado
Android adivinha — e adivinhar errado dá um modelo que roda, não crasha, e responde
lixo. É o bug mais caro deste repositório, porque nenhum teste de código o pega.

## Pré-processamento: um só lugar, e ele é declarado

- A normalização (`/255`, média/desvio) faz parte do **grafo do modelo** sempre que for
  possível — assim ela não pode divergir entre treino e inferência.
- O que não couber no grafo (redimensionar, converter cor, rotação da câmera) mora numa
  única função no Android, testada com um valor conhecido, e está descrito no JSON.
- **Nunca** duas implementações do mesmo pré-processamento. Treino e inferência
  divergindo em uma multiplicação é a causa nº1 de "funcionava no notebook".

## O que não pode acontecer

1. **Vazamento de dados.** Nenhum dado de teste toca no treino — nem em
   normalização, nem em seleção de hiperparâmetro. Split antes de qualquer estatística.
2. **Número sem baseline.** Uma acurácia só significa algo ao lado de um baseline burro
   (classe majoritária, ou um modelo linear). `0,91` sozinho não é resultado.
3. **Modelo não reprodutível.** `seed` fixa e registrada. Se dois treinos dão números
   diferentes, não se sabe se a mudança foi a sua ou o ruído.
4. **Artefato órfão.** `.tflite` commitado sem o script que o gerou no mesmo commit.

## Android

- O intérprete (`ai-edge-litert`) é detalhe de infraestrutura: fica atrás de uma
  interface de domínio (`Classificador`), como o Retrofit fica atrás de um repositório num
  app Android. A UI nunca vê um `Tensor`.
- Carregar o modelo é I/O: acontece fora da main thread, uma vez, e o resultado é
  reaproveitado.
- Medir latência faz parte da entrega, não é otimização prematura: um modelo de 40ms e
  um de 400ms são produtos diferentes.
