"""
Verificação de ambiente — não é teste de feature.

Existe para provar que a cadeia de ferramentas da etapa 1 está de pé **antes** de
qualquer código de treino: interpretador nativo, TensorFlow, e — o que realmente
importa — que a conversão para `.tflite` com quantização int8 funciona nesta máquina, e que
o intérprete LiteRT roda o resultado.

Se este arquivo ficar vermelho, o problema é o ambiente, não o seu modelo.
"""

import platform

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter


def test_given_esta_maquina_when_lemos_o_interpretador_then_ele_e_arm64_nativo():
    # Given
    esperado = "arm64"

    # When
    arch = platform.machine()

    # Then
    assert arch == esperado, (
        f"interpretador {arch} — provavelmente o python3 do Homebrew em /usr/local "
        "(x86_64 sob Rosetta). Use training/.venv, criado com /usr/bin/python3."
    )


def test_given_o_venv_do_repo_when_importamos_tensorflow_then_a_versao_e_a_do_requirements():
    # Given
    esperado = "2.20.0"

    # When
    versao = tf.__version__

    # Then
    assert versao == esperado


def test_given_um_modelo_trivial_when_convertido_para_tflite_int8_then_infere_com_pixel_cru_uint8():
    """
    O ensaio em miniatura da etapa 1: normalização dentro do grafo (Regra 01),
    quantização int8 completa, e entrada `uint8` crua — sem dividir por 255 fora
    do modelo. É o contrato que a spec 0001 declara, testado aqui no menor modelo
    possível.
    """
    # Given
    tf.keras.utils.set_random_seed(42)
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), dtype="float32"),
            tf.keras.layers.Rescaling(1.0 / 255),  # normalização no grafo
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    amostras = np.random.randint(0, 256, size=(16, 28, 28, 1)).astype("float32")

    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    conversor.representative_dataset = lambda: ([amostras[i : i + 1]] for i in range(16))
    conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conversor.inference_input_type = tf.uint8
    conversor.inference_output_type = tf.uint8

    # When
    tflite = conversor.convert()

    # Then
    interprete = Interpreter(model_content=tflite)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    saida = interprete.get_output_details()[0]

    assert entrada["dtype"] == np.uint8
    assert tuple(entrada["shape"]) == (1, 28, 28, 1)
    assert saida["dtype"] == np.uint8
    assert tuple(saida["shape"]) == (1, 10)

    pixel_cru = np.random.randint(0, 256, size=(1, 28, 28, 1)).astype(np.uint8)
    interprete.set_tensor(entrada["index"], pixel_cru)
    interprete.invoke()
    probabilidades = interprete.get_tensor(saida["index"])

    assert probabilidades.shape == (1, 10)
    assert probabilidades.dtype == np.uint8
