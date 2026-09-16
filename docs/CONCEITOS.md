# Conceitos — o mapa para quem nunca treinou um modelo

Se ainda não leu, comece pela [`VISAO-GERAL.md`](VISAO-GERAL.md): o problema e o plano,
sem nenhum destes termos.

Não precisa ler isto de uma vez. É para voltar sempre que aparecer uma palavra estranha
na spec ou nas regras.

## A única ideia que importa no começo

Programação normal:

```
você escreve as regras  →  o programa aplica as regras aos dados
```

Machine learning:

```
você dá exemplos (dados + resposta certa)  →  o programa deriva as regras
```

É a mesma inversão que você já viu em outros lugares: em vez de escrever o `if`, você
mostra mil casos e deixa o processo descobrir qual `if` funciona. O resultado — o
"modelo" — é uma função comum: entra uma imagem, sai uma resposta. O que é incomum é que
**ninguém escreveu os números dentro dela**.

E daí vem a consequência que muda tudo, e que é o motivo de existir a Regra 02:

> **Você não consegue ler o código e saber se o modelo está certo.**
> Só consegue medir. Num app, bug dá crash ou tela errada. Aqui, um modelo errado roda
> liso, não crasha, e devolve respostas plausíveis e erradas.

## Tabela de tradução: o seu mundo → ML

| Termo de ML | O que é, na prática que você já conhece |
|---|---|
| **modelo** | Uma função `(imagem) -> 10 números`. Só isso. O arquivo `.tflite` é essa função serializada |
| **parâmetros** / **pesos** | As constantes dentro dessa função. Uns 100 mil, no nosso caso. Ninguém as escreveu à mão |
| **treinar** | O laço que ajusta essas constantes, aos poucos, para errar menos nos exemplos. É o "build" desta metade do repo — e leva minutos, não segundos |
| **dataset** | A lista de exemplos: imagem + resposta certa ao lado |
| **label** | A resposta certa de um exemplo (`"bota"`) |
| **época** (*epoch*) | Uma passada completa por todos os exemplos. Treinar 10 épocas = ver o dataset inteiro 10 vezes |
| **loss** / função de perda | Um número que mede "o quanto o modelo está errado agora". O treino existe para baixá-lo. Se não baixa, algo está quebrado |
| **inferência** | Chamar a função. É o que o Android vai fazer. Rápido: milissegundos |
| **acurácia** | Percentagem de acertos. `0,91` = acerta 91 em 100 |
| **tensor** | Array multidimensional. `[1, 28, 28, 1]` é "1 imagem, 28×28 pixels, 1 canal de cor". O nome é intimidante e o conceito não é |
| **overfitting** | O modelo decorou o gabarito em vez de aprender. Acerta quase tudo nos exemplos que viu e erra nos novos. É o equivalente exato de um teste que passa porque foi escrito olhando para o bug |
| **split treino / validação / teste** | Três fatias do dataset. Treino: o modelo vê e aprende. Validação: você olha durante o desenvolvimento para decidir coisas. **Teste: você não olha até o fim.** É uma suíte de testes que você se proíbe de rodar, exatamente para não se enganar |
| **baseline** | A versão burra, de propósito. "Chutar sempre a classe mais comum" acerta 10%. Se o seu modelo faz 12%, ele não aprendeu — aprendeu quase nada. Sem baseline, todo número parece bom |
| **quantização** | Trocar os números do modelo de `float32` por `int8`. Como salvar um JPEG com mais compressão: **4× menor e um pouco pior**. É o que torna o modelo viável no celular — e é a fonte do problema central da etapa 1 |
| **matriz de confusão** | Tabela de "o que era" × "o que o modelo disse". Mostra *quais* erros ele comete — uma acurácia de 90% pode esconder uma classe que ele nunca acerta |
| **recall de uma classe** | Das peças que *eram* bota, quantas ele chamou de bota. Se for zero, ele ignora essa classe inteira e a acurácia não te avisa |

## O pipeline da etapa 1, em português

```
1. carregar     60 000 fotos 28×28 de roupas, com a resposta ao lado
2. separar      54 000 para treinar · 6 000 para verificar · 10 000 guardadas (não olhar)
3. treinar      ~3 min de CPU; o loss cai, a acurácia sobe
4. medir        acurácia no conjunto guardado, ao lado do baseline burro
5. exportar     virar .tflite int8 — 4× menor, para caber no celular
6. verificar    o .tflite concorda com o original? ← é aqui que mora o aprendizado
```

O passo 6 é o coração do repositório, e é contra-intuitivo: **o modelo que roda no
celular não é o que você treinou.** A quantização mudou os números dele. Normalmente a
diferença é pequena. Quando não é, você só descobre se comparar — e quase ninguém
compara.

## Por que a etapa 1 não tem nada de Android

Porque misturar as duas metades no primeiro dia significa que, quando der errado, você
não vai saber se o problema é o modelo ou o app. Etapa 1 termina num arquivo em
`models/`. Etapa 2 começa nesse arquivo.

## Se quiser fundamento por fora do repo

Três recursos, em ordem de esforço — nenhum é obrigatório para começar:

1. **[Teachable Machine](https://teachablemachine.withgoogle.com)** — 15 min, no browser.
   Treina um classificador com a sua webcam, sem código. Serve para o conceito "dei
   exemplos, saiu uma função" deixar de ser abstrato.
2. **[3Blue1Brown, série de redes neurais](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)**
   — ~1h de vídeo, visual, sem matemática pesada. É a melhor explicação de *como* o
   treino ajusta os pesos.
3. **[Curso de ML do Google](https://developers.google.com/machine-learning/crash-course)**
   — o mais completo dos três, e o que fala a língua dos termos que você vai encontrar
   nas specs.

## O erro que você vai cometer (todo mundo comete)

Olhar para o conjunto de teste, ver que a acurácia está baixa, mudar algo, e olhar de
novo. Isso parece trabalho honesto e é vazamento de dados: você está usando o conjunto
de teste para tomar decisões, e ele existe justamente para ser a única medida em que
você não influiu. É por isso que a Regra 01 lista isso como o item 1 do "não pode
acontecer" — e por que o CA-01 da spec 0001 é um teste.
