"""
Testes do passo 3.

A nota da rede na prova (CA-04) e o recall por roupa (CA-10) não estão aqui: exigem o
treino completo, e a Regra 02 os coloca no script (`python -m training.src.treino`),
que termina com código de erro se algum falhar.

Aqui fica o que dá para provar rápido: as contas da matriz de confusão com valores
conhecidos, o contrato de entrada da rede, e a reprodutibilidade (CA-09).
"""

import numpy as np
import pytest

from training.src import dados, treino


@pytest.fixture(scope="module")
def tres_montes() -> dados.Dados:
    return dados.carregar()


def test_given_respostas_conhecidas_when_monta_a_matriz_de_confusao_then_cada_erro_cai_na_celula_era_x_disse():
    # Given
    certas = np.array([0, 0, 6, 6, 6])  # duas camisetas, três camisas
    previstas = np.array([0, 6, 6, 0, 0])  # a rede chamou duas camisas de camiseta

    # When
    matriz = treino.matriz_de_confusao(certas, previstas)

    # Then
    assert matriz[0, 0] == 1  # camiseta acertada
    assert matriz[0, 6] == 1  # camiseta chamada de camisa
    assert matriz[6, 6] == 1  # camisa acertada
    assert matriz[6, 0] == 2  # camisa chamada de camiseta
    assert matriz.sum() == 5


def test_given_uma_roupa_que_a_rede_nunca_acerta_when_calcula_o_recall_then_essa_roupa_fica_com_zero():
    # Given
    certas = np.array([0, 0, 6, 6])
    previstas = np.array([0, 0, 0, 0])  # toda camisa virou camiseta
    matriz = treino.matriz_de_confusao(certas, previstas)

    # When
    recall = treino.recall_por_classe(matriz)

    # Then
    assert recall[0] == 1.0
    assert recall[6] == 0.0


def test_given_a_rede_sem_treino_when_recebe_uma_foto_toda_preta_crua_then_devolve_10_probabilidades_que_somam_1():
    # Given
    imagem = np.full((1, 28, 28, 1), 255, dtype=np.uint8)  # pixel cru, sem dividir por 255

    # When
    saida = treino.rede().predict(imagem, verbose=0)

    # Then
    assert saida.shape == (1, 10)
    assert saida.sum() == pytest.approx(1.0, abs=1e-5)


def test_given_a_mesma_semente_when_treina_duas_vezes_then_as_duas_redes_dao_as_mesmas_respostas(
    tres_montes,
):
    """
    CA-09. Com 1 época em vez de 10, para o pytest não levar minutos: o que se prova é
    que a semente trava o sorteio, e isso não depende de quantas épocas se treina.

    Só o limite da spec (< 0,5 ponto) não bastava. Medido em 2026-09-16: **sem**
    semente, duas rodadas divergiram em 418 respostas mas só 0,67 ponto na acurácia, perto
    demais do limite para o teste pegar a falta da semente sempre. Com semente, divergem
    em 0. Comparar resposta por resposta pega a quebra com folga.
    """
    # Given
    validacao = tres_montes.validacao

    # When
    primeira = treino.prever(treino.treinar(tres_montes, epocas=1), validacao.x)
    segunda = treino.prever(treino.treinar(tres_montes, epocas=1), validacao.x)

    # Then
    assert int((primeira != segunda).sum()) == 0
    assert abs(np.mean(primeira == validacao.y) - np.mean(segunda == validacao.y)) < 0.005
