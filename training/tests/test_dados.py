"""
Testes do passo 1 — cobrem o CA-01 da spec 0001.

O teste que importa aqui é o de vazamento (`nenhuma_imagem_aparece_em_dois_montes`).
Os outros afirmam formato; esse afirma **honestidade**: se uma imagem de teste
vazasse para o treino, o modelo acertaria por já a ter visto, e a acurácia final
seria uma mentira que nenhum outro teste pegaria.
"""

import hashlib

import numpy as np
import pytest

from training.src import dados


@pytest.fixture(scope="module")
def tres_montes() -> dados.Dados:
    return dados.carregar()


def _conjunto_de_hashes(split: dados.Split) -> set:
    """
    O conteúdo de cada imagem, resumido num hash, para poder comparar montes por
    conteúdo em vez de por posição.

    Vive fora do teste de propósito: a Regra 02 proíbe laço no corpo do teste, e a
    comparação em si precisa de um. Extrair para um auxiliar mantém o corpo do teste
    plano — três linhas que qualquer pessoa lê sem executar mentalmente.
    """
    return {hashlib.sha1(imagem.tobytes()).hexdigest() for imagem in split.x}


def test_given_dataset_do_keras_when_separa_em_tres_montes_then_os_tamanhos_sao_54000_6000_e_10000(
    tres_montes,
):
    # Given
    esperado = (54_000, 6_000, 10_000)

    # When
    tamanhos = (
        len(tres_montes.treino),
        len(tres_montes.validacao),
        len(tres_montes.teste),
    )

    # Then
    assert tamanhos == esperado


def test_given_os_tres_montes_when_compara_o_conteudo_das_imagens_then_nenhuma_imagem_aparece_em_dois_montes(
    tres_montes,
):
    # Given
    treino = _conjunto_de_hashes(tres_montes.treino)
    validacao = _conjunto_de_hashes(tres_montes.validacao)
    teste = _conjunto_de_hashes(tres_montes.teste)

    # When
    vazamentos = (treino & validacao) | (treino & teste) | (validacao & teste)

    # Then
    assert vazamentos == set()


def test_given_a_validacao_when_compara_com_o_fim_do_treino_original_then_ela_saiu_do_treino_e_nao_do_teste(
    tres_montes,
):
    # Given
    (x_treino_original, _), _ = dados.tf.keras.datasets.fashion_mnist.load_data()
    ultimas_do_treino = x_treino_original[-dados.TAMANHO_VALIDACAO :]

    # When
    validacao_sem_canal = tres_montes.validacao.x.reshape(ultimas_do_treino.shape)

    # Then
    assert np.array_equal(validacao_sem_canal, ultimas_do_treino)


def test_given_o_monte_de_treino_when_olha_a_forma_das_imagens_then_tem_o_canal_de_cor_esperado_pelo_modelo(
    tres_montes,
):
    # Given
    esperado = (54_000, 28, 28, 1)

    # When
    forma = tres_montes.treino.x.shape

    # Then
    assert forma == esperado


def test_given_as_imagens_carregadas_when_olha_os_valores_dos_pixels_then_sao_uint8_crus_de_0_a_255(
    tres_montes,
):
    # Given
    imagens = tres_montes.treino.x

    # When
    dtype, minimo, maximo = imagens.dtype, imagens.min(), imagens.max()

    # Then
    assert dtype == np.uint8
    assert (minimo, maximo) == (0, 255)


def test_given_as_dez_classes_when_conta_as_labels_do_treino_then_todas_aparecem_e_ficam_proximas_de_5400(
    tres_montes,
):
    # Given
    esperado_por_classe = 5_400

    # When
    contagem = np.bincount(tres_montes.treino.y, minlength=len(dados.LABELS))

    # Then
    assert len(contagem) == 10
    assert contagem.min() > 0
    assert np.abs(contagem - esperado_por_classe).max() < 100
