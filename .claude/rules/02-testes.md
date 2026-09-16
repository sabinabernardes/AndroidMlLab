# Regra 02 — Testes

## Formato obrigatório: Given / When / Then

Os três blocos comentados, e o nome descreve os três.
Em Python, `snake_case` (não há backticks), em português:

```python
def test_given_imagem_toda_preta_when_classifica_then_devolve_10_probabilidades_que_somam_1():
    # Given
    imagem = np.zeros((28, 28, 1), dtype=np.uint8)

    # When
    saida = classificar(imagem)

    # Then
    assert saida.shape == (10,)
    assert saida.sum() == pytest.approx(1.0, abs=1e-3)
```

- Um `when` por teste. Sem `if`/`for` no corpo.
- O `then` afirma **comportamento observável**, não implementação.

## O que testar em ML — e é diferente de um app

| Alvo | Testar | Como |
|---|---|---|
| Carregamento de dados | Shapes, dtype, intervalo de valores, **tamanho dos splits e que não se intersectam** | pytest puro |
| Pré-processamento | Um valor conhecido entra, valor conhecido sai. Bordas: 0 e 255 | pytest puro |
| Export `.tflite` | Shape e dtype de entrada/saída batem com o JSON de metadata | pytest + intérprete |
| **Invariância Keras ↔ tflite** | Mesmo lote de entrada nos dois → mesma classe predita, e a acurácia não cai mais que o limite declarado na spec | pytest, obrigatório |
| Métrica do modelo | Acurácia no conjunto de teste **acima do baseline declarado na spec** | script de avaliação |
| Android: `Classificador` | Saída para uma imagem empacotada de classe conhecida | teste instrumentado |

## O que NÃO testar

- Que o Keras treina. Que o TensorFlow multiplica matrizes.
- Que a acurácia é exatamente `0.9137`. Número exato de treino é frágil por natureza —
  afirme um **piso** (`>= 0.88`), nunca uma igualdade.
- Que o treino "roda sem erro" sem nenhuma asserção sobre o resultado.

## A armadilha específica deste repositório

Em app comum, teste verde e comportamento errado é raro. Aqui é o caso **normal**: o
pipeline inteiro pode estar verde com o modelo respondendo lixo, porque o erro está nos
números, não no fluxo. Daí as duas exigências que não são negociáveis:

1. **Baseline.** Toda métrica na spec vem com o número do modelo burro ao lado. Se o seu
   modelo não bate a classe majoritária, ele não aprendeu nada — e o teste de acurácia
   passaria feliz num limite mal escolhido.
2. **Teste de invariância no export.** Quantizar é perder precisão de propósito. Sem
   comparar Keras e `.tflite` no mesmo lote, a perda só aparece no celular.

## Definição de pronto

- [ ] Todo critério de aceite da spec tem ao menos um teste que o cobre.
- [ ] `pytest` verde.
- [ ] A métrica declarada foi batida, e o baseline está registrado ao lado dela.
- [ ] O `.tflite` e seu JSON estão no mesmo commit do script que os gerou.
- [ ] Você quebrou de propósito um passo do pré-processamento e viu o teste ficar vermelho.
