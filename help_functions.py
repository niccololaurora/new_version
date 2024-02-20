import numpy as np
import tensorflow as tf


def create_target1(nclasses):
    if nclasses == 2:
        targets = tf.constant(
            [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
        )

    if nclasses == 3:
        targets = tf.constant(
            [
                [1, 0],
                [np.cos(np.pi / 3), np.sin(np.pi / 3)],
                [np.cos(np.pi / 3), -np.sin(np.pi / 3)],
            ],
            dtype=tf.complex64,
        )

    if nclasses == 4:
        alpha = np.arctan(1 / np.sqrt(3))
        theta = np.pi / 2 - alpha
        targets = tf.constant(
            [
                np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), -np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)], dtype="complex"),
            ]
        )
    if nclasses == 6:
        theta = np.pi
        targets = tf.constant(
            [
                np.array([1, 0], dtype="complex"),
                np.array([0, 1], dtype="complex"),
                np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), -np.sin(theta / 2)], dtype="complex"),
                np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)], dtype="complex"),
            ],
        )

    if nclasses == 10:
        targets = 0

    return targets


def create_target(nclasses):
    if nclasses == 2:
        targets = [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
    if nclasses == 3:
        targets = [
            np.array([1, 0], dtype="complex"),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)], dtype="complex"),
            np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)], dtype="complex"),
        ]
    if nclasses == 4:
        alpha = np.arctan(1 / np.sqrt(3))
        theta = np.pi / 2 - alpha
        targets = [
            np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), -np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)], dtype="complex"),
        ]

    if nclasses == 6:
        theta = np.pi
        targets = [
            np.array([1, 0], dtype="complex"),
            np.array([0, 1], dtype="complex"),
            np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), -np.sin(theta / 2)], dtype="complex"),
            np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)], dtype="complex"),
        ]

    if nclasses == 10:
        targets = 0

    return targets


def fidelity(state1, state2):
    """
    Args: two vectors
    Output: inner product of the two vectors **2
    """
    norm = tf.math.abs(tf.reduce_sum(tf.math.conj(state1) * state2))
    return norm


def number_params(n_embed_params, nqubits, pooling):

    if pooling != "no":
        return 2 * nqubits + n_embed_params
    else:
        return n_embed_params
