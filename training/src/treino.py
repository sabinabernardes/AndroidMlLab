"""
Passo 3 de 5 — treinar a rede, o "aluno bom".

A diferença para a regressão logística do passo 2: aquela dava um peso para cada pixel,
na posição fixa dele. Esta rede tem **camadas convolucionais**, que passam uma janelinha
3×3 por toda a foto procurando padrões — uma borda, uma curva, uma linha vertical —
em qualquer lugar da imagem. Empilhando duas, ela passa a enxergar **formato**: gola,
manga, cano de bota. É o que falta à logística (ver `docs/VISAO-GERAL.md`).

Ao fim, a rede faz a prova (o monte de teste) uma única vez e o resultado é comparado
com os baselines do passo 2. Rodar este script de novo é normal; o que não pode é mudar
a rede *por causa* da nota da prova — para decidir, olha-se a validação (Regra 01).

Critérios da spec 0001 cobertos aqui: CA-04 (nota ≥ 88% e acima da logística) e CA-10
(nenhuma roupa ignorada). O CA-09 (reprodutibilidade) está em `tests/test_treino.py`.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # desenha direto em arquivo, sem abrir janela do Python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from training.src import baseline, dados, semente

EPOCAS = 10
META_ACURACIA = 0.88  # spec 0001, CA-04
PASTA_RUNS = baseline.PASTA_RUNS
ARQUIVO_MODELO = PASTA_RUNS / "fashion_mnist_cnn.keras"  # o passo 4 lê daqui
ARQUIVO_RESULTADO = PASTA_RUNS / "treino.json"
ARQUIVO_MATRIZ = PASTA_RUNS / "matriz_confusao.png"


def rede() -> tf.keras.Model:
    """A arquitetura declarada na spec 0001, seção "Modelo e números"."""
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), dtype="float32"),
            # A divisão por 255 mora aqui dentro (Regra 01): o Android entrega pixel cru.
            tf.keras.layers.Rescaling(1.0 / 255),
            # Procura 32 padrões pequenos (bordas, curvas) em toda a foto...
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            # ...e resume cada região 2×2 no seu valor mais forte: a foto "encolhe".
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Sobre esse resumo, procura 64 padrões maiores (combinações dos primeiros).
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            # Desliga 30% dos neurônios a cada passo do treino, ao acaso: obriga a rede a
            # não depender de nenhum detalhe isolado. É um freio contra decorar.
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(dados.LABELS), activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def treinar(montes: dados.Dados, epocas: int = EPOCAS, verbose: int = 0) -> tf.keras.Model:
    """Treina no treino, acompanha na validação. O teste não entra aqui."""
    semente.fixar()
    modelo = rede()
    modelo.fit(
        montes.treino.x,
        montes.treino.y,
        validation_data=(montes.validacao.x, montes.validacao.y),
        epochs=epocas,
        batch_size=128,
        verbose=verbose,
    )
    return modelo


def prever(modelo: tf.keras.Model, imagens: np.ndarray) -> np.ndarray:
    """A classe que a rede escolhe para cada imagem: a de maior probabilidade."""
    return modelo.predict(imagens, batch_size=1024, verbose=0).argmax(axis=1)


def matriz_de_confusao(certas: np.ndarray, previstas: np.ndarray) -> np.ndarray:
    """
    Tabela 10×10: linha = o que a roupa era, coluna = o que a rede disse.

    A diagonal são os acertos. Todo o resto é um tipo específico de erro — por exemplo,
    linha "camisa", coluna "camiseta" conta quantas camisas a rede chamou de camiseta.
    """
    classes = len(dados.LABELS)
    matriz = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matriz, (certas, previstas), 1)
    return matriz


def recall_por_classe(matriz: np.ndarray) -> np.ndarray:
    """
    De cada roupa, a fração que a rede acertou. Zero numa classe = a rede a ignora por
    completo — e a acurácia geral pode estar alta mesmo assim (CA-10).
    """
    return np.diag(matriz) / matriz.sum(axis=1)


def maiores_confusoes(matriz: np.ndarray, quantas: int) -> list:
    """Os erros mais frequentes, como (era, disse, vezes), do maior para o menor."""
    erros = matriz.copy()
    np.fill_diagonal(erros, 0)
    posicoes = np.argsort(erros, axis=None)[::-1][:quantas]
    return [
        (dados.LABELS[era], dados.LABELS[disse], int(erros[era, disse]))
        for era, disse in zip(*np.unravel_index(posicoes, erros.shape))
    ]


def _desenhar_matriz(matriz: np.ndarray, destino: Path) -> None:
    figura, eixo = plt.subplots(figsize=(10, 9))
    eixo.imshow(matriz, cmap="Blues")
    for era in range(matriz.shape[0]):
        for disse in range(matriz.shape[1]):
            valor = int(matriz[era, disse])
            eixo.text(
                disse,
                era,
                valor,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if valor > matriz.max() / 2 else "black",
            )
    eixo.set_xticks(range(len(dados.LABELS)), dados.LABELS, rotation=45, ha="right")
    eixo.set_yticks(range(len(dados.LABELS)), dados.LABELS)
    eixo.set_xlabel("o que a rede disse")
    eixo.set_ylabel("o que a roupa era")
    eixo.set_title("Matriz de confusão no teste — diagonal = acertos (1000 fotos por linha)")
    figura.tight_layout()
    figura.savefig(destino, dpi=110)
    plt.close(figura)


def _ler_baseline() -> dict:
    if not baseline.ARQUIVO_RESULTADO.exists():
        sys.exit(
            "Falta o resultado do passo 2. Rode antes:\n"
            "  training/.venv/bin/python -m training.src.baseline"
        )
    return json.loads(baseline.ARQUIVO_RESULTADO.read_text())


def main() -> None:
    referencia = _ler_baseline()
    logistica = referencia["regressao_logistica"]["acuracia_teste"]
    majoritaria = referencia["classe_majoritaria"]["acuracia_teste"]
    montes = dados.carregar()

    print(f"\nTreinando a rede ({EPOCAS} épocas, ~2 min de CPU)...")
    print(
        "Compare com o passo 2: a logística terminou com val_accuracy 0.8535.\n"
        "Fique de olho também na distância entre 'accuracy' e 'val_accuracy':\n"
        "se a primeira disparar e a segunda parar, a rede começou a decorar.\n"
    )
    modelo = treinar(montes, verbose=2)
    PASTA_RUNS.mkdir(exist_ok=True)
    modelo.save(ARQUIVO_MODELO)

    # A prova: uma única passada pelo teste, reaproveitada para tudo abaixo.
    previstas = prever(modelo, montes.teste.x)
    acuracia_teste = float(np.mean(previstas == montes.teste.y))
    matriz = matriz_de_confusao(montes.teste.y, previstas)
    recall = recall_por_classe(matriz)
    _desenhar_matriz(matriz, ARQUIVO_MATRIZ)

    ca04 = acuracia_teste >= META_ACURACIA and acuracia_teste > logistica
    ca10 = bool(recall.min() > 0)

    ARQUIVO_RESULTADO.write_text(
        json.dumps(
            {
                "semente": semente.SEMENTE,
                "epocas": EPOCAS,
                "acuracia_teste": round(acuracia_teste, 4),
                "baseline_logistica": logistica,
                "baseline_majoritaria": majoritaria,
                "recall_por_classe": {
                    nome: round(float(valor), 4) for nome, valor in zip(dados.LABELS, recall)
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    marca = {True: "✅", False: "❌"}
    linhas = [
        "",
        "A prova — 10 000 fotos que nenhum modelo viu",
        "-" * 56,
        f"Classe majoritária    {majoritaria:.2%}",
        f"Regressão logística   {logistica:.2%}",
        f"Rede (este passo)     {acuracia_teste:.2%}   ({acuracia_teste - logistica:+.2%} sobre a logística)",
        "-" * 56,
        "",
        "Acerto por roupa (recall)",
    ]
    for nome, valor in sorted(zip(dados.LABELS, recall), key=lambda par: par[1]):
        linhas.append(f"  {nome:<10} {valor:.1%}  {'█' * int(valor * 30)}")
    linhas += ["", "Os erros mais comuns"]
    for era, disse, vezes in maiores_confusoes(matriz, 5):
        linhas.append(f"  {vezes:>4}× era {era}, a rede disse {disse}")
    linhas += [
        "",
        f"{marca[ca04]} CA-04  acurácia >= {META_ACURACIA:.0%} e acima da logística",
        f"{marca[ca10]} CA-10  nenhuma roupa com recall zero",
        "",
        f"Modelo salvo em training/runs/{ARQUIVO_MODELO.name} (o passo 4 lê daqui)",
        f"Matriz de confusão em training/runs/{ARQUIVO_MATRIZ.name}",
        "",
    ]
    print("\n".join(linhas))

    if platform.system() == "Darwin":
        subprocess.run(["open", str(ARQUIVO_MATRIZ)], check=False)
    if not (ca04 and ca10):
        sys.exit(1)


if __name__ == "__main__":
    main()
