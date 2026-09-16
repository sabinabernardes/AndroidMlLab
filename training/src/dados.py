"""
Passo 1 de 5 — carregar os dados e separá-los.

É o passo mais chato e o que mais estraga resultado de ML. Tudo o que ele faz:
baixar 70 000 fotos de roupas, dar a elas a forma que o modelo espera, e cortá-las
em três montes que não se misturam.

O corte é o ponto delicado. Ver `docs/CONCEITOS.md`, seção "split treino /
validação / teste": o monte de teste é aquele que você se proíbe de olhar até o fim.
"""

from typing import NamedTuple

import numpy as np
import tensorflow as tf

# Na ordem do dataset — o índice é o que o modelo devolve, o nome é para humanos.
LABELS = (
    "camiseta",
    "calça",
    "pulôver",
    "vestido",
    "casaco",
    "sandália",
    "camisa",
    "tênis",
    "bolsa",
    "bota",
)

# Quantas imagens do fim do monte de treino viram validação (spec 0001, seção Dados).
TAMANHO_VALIDACAO = 6_000


class Split(NamedTuple):
    """Um monte de exemplos: as imagens e as respostas certas."""

    x: np.ndarray  # (N, 28, 28, 1) uint8 — pixels crus, de 0 a 255
    y: np.ndarray  # (N,) uint8 — o índice da label certa, de 0 a 9

    def __len__(self) -> int:
        return len(self.y)


class Dados(NamedTuple):
    treino: Split
    validacao: Split
    teste: Split


def _com_canal(imagens: np.ndarray) -> np.ndarray:
    """
    (N, 28, 28) -> (N, 28, 28, 1).

    A dimensão extra é o "canal de cor". Uma foto colorida teria 3 (vermelho, verde,
    azul); estas são em tons de cinza, então têm 1. A camada convolucional exige que
    essa dimensão exista, mesmo valendo 1.
    """
    return imagens.reshape((*imagens.shape, 1))


def carregar() -> Dados:
    """
    Devolve os três montes. Na primeira execução baixa ~30 MB para ~/.keras/datasets;
    depois é instantâneo.

    Note o que NÃO acontece aqui: nenhuma normalização, nenhuma média, nenhum desvio
    padrão. Os pixels saem crus, de 0 a 255. A divisão por 255 é uma camada dentro do
    modelo (Regra 01) — assim é impossível o treino e o Android discordarem.
    """
    (x_treino_completo, y_treino_completo), (x_teste, y_teste) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # O corte: as últimas 6 000 do treino viram validação. O monte de teste não é
    # tocado — é por isso que o corte sai daqui, e não de lá.
    corte = len(x_treino_completo) - TAMANHO_VALIDACAO

    return Dados(
        treino=Split(_com_canal(x_treino_completo[:corte]), y_treino_completo[:corte]),
        validacao=Split(_com_canal(x_treino_completo[corte:]), y_treino_completo[corte:]),
        teste=Split(_com_canal(x_teste), y_teste),
    )


def _resumo(dados: Dados) -> str:
    linhas = ["", "Os três montes", "-" * 52]
    for nome, split in (
        ("treino", dados.treino),
        ("validação", dados.validacao),
        ("teste (não olhar)", dados.teste),
    ):
        linhas.append(
            f"{nome:<20} {len(split):>6} imagens   "
            f"{split.x.shape}  {split.x.dtype}"
        )

    linhas += ["", "Quantas imagens de cada classe, no treino", "-" * 52]
    contagem = np.bincount(dados.treino.y, minlength=len(LABELS))
    for indice, nome in enumerate(LABELS):
        linhas.append(f"  {indice}  {nome:<12} {contagem[indice]:>6}")

    linhas += [
        "",
        f"Pixels: de {dados.treino.x.min()} a {dados.treino.x.max()} "
        f"(crus, sem normalização — ela mora no modelo)",
        "",
    ]
    return "\n".join(linhas)


if __name__ == "__main__":
    print(_resumo(carregar()))
