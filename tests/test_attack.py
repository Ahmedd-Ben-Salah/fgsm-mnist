import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack import fgsm, input_gradient


def tiny_model():
    tf.random.set_seed(0)
    return tf.keras.Sequential([tf.keras.Input(shape=(28, 28)),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(8, activation="relu"),
                                tf.keras.layers.Dense(10)])


def sample():
    rng = np.random.default_rng(0)
    image = rng.random((28, 28)).astype("float32")
    label = np.int64(3)
    return image, label


def test_gradient_has_image_shape_and_is_nonzero():
    model = tiny_model()
    image, label = sample()
    grad = input_gradient(model, image, label)
    assert tuple(grad.shape) == (1, 28, 28)
    assert float(tf.reduce_max(tf.abs(grad))) > 0.0


def test_zero_epsilon_leaves_the_image_unchanged():
    model = tiny_model()
    image, label = sample()
    adv = fgsm(model, image, label, 0.0)
    assert float(tf.reduce_max(tf.abs(adv[0] - image))) == 0.0


def test_perturbation_is_bounded_by_epsilon_and_stays_in_range():
    model = tiny_model()
    image, label = sample()
    epsilon = 0.1
    adv = fgsm(model, image, label, epsilon)
    assert float(tf.reduce_max(tf.abs(adv[0] - image))) <= epsilon + 1e-6
    assert float(tf.reduce_min(adv)) >= 0.0
    assert float(tf.reduce_max(adv)) <= 1.0
