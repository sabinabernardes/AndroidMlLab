# Visão geral — o que estamos fazendo, e por quê

Comece por aqui. Sem jargão: os termos técnicos ficam para o
[`CONCEITOS.md`](CONCEITOS.md), que é para consultar depois.

## O problema que estamos resolvendo

Imagine um app: você aponta a câmera para uma roupa e a tela mostra **"isso é uma bota"**.

Como programadora Android, você escreveria isso assim:

```kotlin
fun qualRoupa(foto: Bitmap): String {
    if (???) return "bota"
    if (???) return "camiseta"
    ...
}
```

Só que ninguém consegue escrever esses `if`. Você **reconhece** uma bota na hora, mas
não consegue explicar em código o que faz uma foto ser uma bota.

**Machine learning é o jeito de criar essa função sem escrever os `if`.** Em vez de
regras, você entrega um monte de fotos já etiquetadas ("esta é bota", "esta é
camiseta"), e o computador ajusta a função sozinho até ela acertar.

Este repositório faz uma versão pequena desse app para aprender o caminho inteiro: fotos
minúsculas (28×28, preto e branco) de 10 tipos de roupa. Para vê-las:

```bash
training/.venv/bin/python -m training.src.ver_fotos
```

## O plano, como se fosse ensinar um aluno

Pense no computador como um **aluno** que precisa aprender a reconhecer roupas.

| Passo | O que é, em português | No código |
|---|---|---|
| **1. Separar o material** | Temos 70 mil fotos etiquetadas. 60 mil viram **material de estudo** e 10 mil ficam **trancadas na gaveta para a prova final**. O aluno nunca pode ver a prova antes, senão ele decora as respostas e a nota vira mentira. | `training/src/dados.py` |
| **2. Medir alunos fracos** | Antes de treinar o aluno bom, vemos quanto tiram alunos ruins. O **preguiçoso** chuta "tênis" em tudo e tira **10%**. O **esforçado mas limitado** olha pixel por pixel, sem entender formato, e tira **84%**. | `training/src/baseline.py` |
| **3. Treinar o aluno bom** | Um modelo que enxerga **formato** (gola, manga, cano). A meta é tirar **88% ou mais**. Se tirar 85%, ele não foi muito melhor que o limitado, e não valeu a pena. | `training/src/treino.py` |
| **4. Encolher para caber no celular** | O modelo treinado é "pesado". Comprimimos para um arquivo `.tflite` 4× menor, como exportar uma imagem com mais compressão: fica menor e um pouquinho pior. | `training/src/export.py` |
| **5. Conferir se a compressão estragou** | Damos a mesma prova ao modelo original e ao comprimido. Se as respostas forem quase iguais, o arquivo pode ir para o celular. | `training/src/verify.py` |

Depois disso vem a **etapa 2**: um app Android que carrega esse `.tflite` e mostra
"bota" na tela. Essa é a parte em que você já é especialista.

Qual passo está feito fica no `CLAUDE.md`, seção "Estado atual". Os detalhes e números
exatos de cada passo estão na [`spec 0001`](../specs/0001-classificador-fashion-mnist.md).

## Por que o passo 2 existe

Suponha que no passo 3 o modelo tire **87%**. Isso é bom?

Sozinho, não dá para saber. Mas o passo 2 mostrou que um aluno que só olha pixels já
tira 84%. Então 87% é **pouco**: um modelo muito mais complexo ganhou só 3 pontos. Sem o
passo 2, 87% pareceria ótimo.

É como medir o tempo de abertura de um app: "800ms" só significa algo se você sabe
quanto os outros apps levam.
