"""
Testes do script de visualização. Não cobre critério de aceite da spec — só garante
que a foto mostrada ao lado de uma etiqueta é mesmo daquela etiqueta.
"""

import numpy as np

from training.src import ver_fotos


def test_given_labels_misturadas_when_escolhe_duas_fotos_por_classe_then_cada_classe_recebe_as_suas_primeiras_posicoes():
    # Given
    labels = np.array([3, 1, 3, 1, 3, 9], dtype=np.uint8)

    # When
    escolhidas = ver_fotos.indices_por_classe(labels, por_classe=2)

    # Then
    assert escolhidas[3].tolist() == [0, 2]
    assert escolhidas[1].tolist() == [1, 3]
    assert escolhidas[9].tolist() == [5]
    assert escolhidas[0].tolist() == []
