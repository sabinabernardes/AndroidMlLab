"""
Testes do passo 2 — cobrem o CA-02 e o CA-03 da spec 0001.

A logística é avaliada na **validação**, não no teste: o pytest roda muitas vezes, e
cada rodada seria uma olhada no monte que só pode ser olhado uma vez.
"""

import numpy as np
import pytest

from training.src import baseline, dados


@pytest.fixture(scope="module")
def tres_montes() -> dados.Dados:
    return dados.carregar()


def _split(labels: list) -> dados.Split:
    """Um monte de mentira: só as labels importam para a classe majoritária."""
    y = np.array(labels, dtype=np.uint8)
    return dados.Split(np.zeros((len(y), 28, 28, 1), dtype=np.uint8), y)


def test_given_treino_com_mais_calcas_when_avalia_a_classe_majoritaria_then_acerta_so_as_calcas_do_monte_avaliado():
    # Given
    treino = _split([1, 1, 1, 9])  # três calças e uma bota
    avaliado = _split([1, 9, 9, 9])  # só uma calça

    # When
    acuracia = baseline.acuracia_classe_majoritaria(treino, avaliado)

    # Then
    assert acuracia == 0.25


def test_given_o_dataset_balanceado_when_avalia_a_classe_majoritaria_no_teste_then_acerta_perto_de_10_por_cento(
    tres_montes,
):
    # Given
    treino, teste = tres_montes.treino, tres_montes.teste

    # When
    acuracia = baseline.acuracia_classe_majoritaria(treino, teste)

    # Then
    assert acuracia == pytest.approx(0.10, abs=0.005)


def test_given_a_regressao_logistica_treinada_when_avalia_na_validacao_then_supera_de_longe_a_classe_majoritaria(
    tres_montes,
):
    # Given
    modelo = baseline.treinar_regressao_logistica(tres_montes, epocas=3)

    # When
    acuracia = baseline.acuracia(modelo, tres_montes.validacao)

    # Then
    assert acuracia >= 0.78
