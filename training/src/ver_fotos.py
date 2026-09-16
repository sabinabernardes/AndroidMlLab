"""
Olhar os dados antes de treinar — não é um passo da spec, é um hábito.

Gera duas imagens em `training/runs/` e as abre:

1. `fotos.png` — 8 fotos de cada uma das 10 roupas, com a etiqueta ao lado. É o
   material de estudo do modelo, do jeito que ele recebe.
2. `fotos_numeros.png` — uma única foto ampliada, com o valor de cada pixel escrito
   em cima. É o que o modelo "vê" de verdade: não uma bota, e sim 784 números de 0 a 255.

Só usa o monte de **treino**. O de teste continua trancado (ver `docs/VISAO-GERAL.md`).
"""

import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # desenha direto em arquivo, sem abrir janela do Python
import matplotlib.pyplot as plt
import numpy as np

from training.src import dados

FOTOS_POR_CLASSE = 8
PASTA_RUNS = Path(__file__).resolve().parents[1] / "runs"
CLASSE_AMPLIADA = dados.LABELS.index("bota")


def indices_por_classe(labels: np.ndarray, por_classe: int) -> dict:
    """As primeiras `por_classe` posições de cada label: {label: array de posições}."""
    return {
        classe: np.flatnonzero(labels == classe)[:por_classe]
        for classe in range(len(dados.LABELS))
    }


def _desenhar_grade(treino: dados.Split, destino: Path) -> None:
    escolhidas = indices_por_classe(treino.y, FOTOS_POR_CLASSE)
    figura, eixos = plt.subplots(
        len(dados.LABELS), FOTOS_POR_CLASSE, figsize=(FOTOS_POR_CLASSE * 1.1, 12)
    )
    for classe, posicoes in escolhidas.items():
        for coluna, posicao in enumerate(posicoes):
            eixo = eixos[classe, coluna]
            eixo.imshow(treino.x[posicao, :, :, 0], cmap="gray_r", vmin=0, vmax=255)
            eixo.set_xticks([])
            eixo.set_yticks([])
        eixos[classe, 0].set_ylabel(
            f"{classe} · {dados.LABELS[classe]}", rotation=0, ha="right", va="center"
        )
    figura.suptitle("Fashion-MNIST — 8 fotos de cada roupa (monte de treino)")
    figura.tight_layout()
    figura.savefig(destino, dpi=120)
    plt.close(figura)


def _desenhar_numeros(treino: dados.Split, destino: Path) -> None:
    posicao = indices_por_classe(treino.y, 1)[CLASSE_AMPLIADA][0]
    pixels = treino.x[posicao, :, :, 0]

    figura, eixo = plt.subplots(figsize=(14, 14))
    eixo.imshow(pixels, cmap="gray_r", vmin=0, vmax=255)
    for linha in range(pixels.shape[0]):
        for coluna in range(pixels.shape[1]):
            valor = int(pixels[linha, coluna])
            eixo.text(
                coluna,
                linha,
                valor,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if valor > 128 else "black",
            )
    eixo.set_xticks([])
    eixo.set_yticks([])
    eixo.set_title(
        f'O que o modelo recebe quando a foto é uma "{dados.LABELS[CLASSE_AMPLIADA]}": '
        "28 × 28 = 784 números (0 = fundo, 255 = mais escuro)"
    )
    figura.tight_layout()
    figura.savefig(destino, dpi=100)
    plt.close(figura)


def main() -> None:
    treino = dados.carregar().treino
    PASTA_RUNS.mkdir(exist_ok=True)
    grade, numeros = PASTA_RUNS / "fotos.png", PASTA_RUNS / "fotos_numeros.png"

    _desenhar_grade(treino, grade)
    _desenhar_numeros(treino, numeros)

    print(f"\nSalvo em training/runs/{grade.name} e training/runs/{numeros.name}\n")
    if platform.system() == "Darwin":
        subprocess.run(["open", str(grade), str(numeros)], check=False)


if __name__ == "__main__":
    main()
