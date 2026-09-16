"""
Passo 2 de 5 — os baselines: quanto se acerta sem aprender quase nada.

Dois modelos burros, de propósito:

1. **Classe majoritária.** Olha o monte de treino, vê qual roupa aparece mais, e
   responde sempre essa. Não olha a imagem. É o piso absoluto: um modelo que não bate
   isto não aprendeu nada.
2. **Regressão logística.** O modelo "de verdade" mais simples que existe: uma única
   camada, que dá um peso para cada pixel e soma. Não enxerga forma, só "pixel escuro
   aqui costuma ser calça". É o número que a rede do passo 3 precisa superar para ter
   valido a pena (spec 0001, CA-03).

O monte de teste é usado só no `__main__`, uma vez, para reportar. Nada aqui é escolhido
olhando para ele.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from training.src import dados, semente

EPOCAS_LOGISTICA = 10
PASTA_RUNS = Path(__file__).resolve().parents[1] / "runs"
ARQUIVO_RESULTADO = PASTA_RUNS / "baseline.json"


def classe_majoritaria(treino: dados.Split) -> int:
    """A label mais frequente no treino. Só no treino — escolhê-la no teste seria vazamento."""
    return int(np.bincount(treino.y).argmax())


def acuracia_classe_majoritaria(treino: dados.Split, avaliado: dados.Split) -> float:
    """Fração de `avaliado` que acertaria quem sempre responde a classe majoritária do treino."""
    return float(np.mean(avaliado.y == classe_majoritaria(treino)))


def regressao_logistica() -> tf.keras.Model:
    """
    Pixel cru entra, 10 probabilidades saem, e no meio só uma multiplicação e uma soma.

    O `Rescaling` é a mesma normalização no grafo que a rede vai usar (Regra 01): a
    comparação entre os dois só é justa se os dois recebem a mesma entrada.
    """
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), dtype="float32"),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Flatten(),  # 28×28 -> 784 números numa fila
            tf.keras.layers.Dense(len(dados.LABELS), activation="softmax"),
        ],
        name="regressao_logistica",
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def treinar_regressao_logistica(
    montes: dados.Dados, epocas: int = EPOCAS_LOGISTICA, verbose: int = 0
) -> tf.keras.Model:
    """Treina no treino, acompanha na validação. O teste não entra aqui."""
    semente.fixar()
    modelo = regressao_logistica()
    modelo.fit(
        montes.treino.x,
        montes.treino.y,
        validation_data=(montes.validacao.x, montes.validacao.y),
        epochs=epocas,
        batch_size=128,
        verbose=verbose,
    )
    return modelo


def acuracia(modelo: tf.keras.Model, avaliado: dados.Split) -> float:
    _, valor = modelo.evaluate(avaliado.x, avaliado.y, verbose=0)
    return float(valor)


def main() -> None:
    montes = dados.carregar()

    classe = classe_majoritaria(montes.treino)
    majoritaria_teste = acuracia_classe_majoritaria(montes.treino, montes.teste)

    print(f"\nTreinando a regressão logística ({EPOCAS_LOGISTICA} épocas)...")
    print("Repare no 'loss' caindo e no 'val_accuracy' subindo a cada época.\n")
    logistica = treinar_regressao_logistica(montes, verbose=2)
    logistica_validacao = acuracia(logistica, montes.validacao)
    logistica_teste = acuracia(logistica, montes.teste)  # a única olhada no teste

    resultado = {
        "semente": semente.SEMENTE,
        "classe_majoritaria": {
            "classe": dados.LABELS[classe],
            "acuracia_teste": round(majoritaria_teste, 4),
        },
        "regressao_logistica": {
            "epocas": EPOCAS_LOGISTICA,
            "acuracia_validacao": round(logistica_validacao, 4),
            "acuracia_teste": round(logistica_teste, 4),
        },
    }
    PASTA_RUNS.mkdir(exist_ok=True)
    ARQUIVO_RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))

    print(
        f"""
Baselines — o que a rede do passo 3 precisa bater
{"-" * 56}
Classe majoritária   sempre "{dados.LABELS[classe]}"        teste {majoritaria_teste:.2%}
Regressão logística  1 camada, sem ver forma   teste {logistica_teste:.2%}
{"-" * 56}
Meta da spec para a rede: >= 88% e acima da logística.

Salvo em training/runs/{ARQUIVO_RESULTADO.name} (o passo 3 lê daqui)
"""
    )


if __name__ == "__main__":
    main()
