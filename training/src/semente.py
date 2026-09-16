"""
A semente fixa da spec 0001 (Regra 01, item 3: modelo reprodutível).

Treinar começa com números sorteados — os pesos iniciais, a ordem em que os exemplos
são vistos. Sem fixar o sorteio, dois treinos idênticos dão números diferentes, e fica
impossível saber se uma melhora foi sua ou sorte. É o mesmo motivo de um teste não
depender de `System.currentTimeMillis()`.
"""

import tensorflow as tf

SEMENTE = 42


def fixar() -> None:
    """Chame antes de construir qualquer modelo."""
    tf.keras.utils.set_random_seed(SEMENTE)  # Python, NumPy e TensorFlow de uma vez
    tf.config.experimental.enable_op_determinism()
