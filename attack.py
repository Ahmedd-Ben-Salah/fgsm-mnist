import tensorflow as tf


def input_gradient(model, image, label):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(image.shape) == 2:
        image = image[None, ...]
    label = tf.convert_to_tensor(label)
    if len(label.shape) == 0:
        label = label[None]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        tape.watch(image)
        logits = model(image)
        loss = loss_fn(label, logits)
    return tape.gradient(loss, image)


def fgsm(model, image, label, epsilon):
    grad = input_gradient(model, image, label)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(image.shape) == 2:
        image = image[None, ...]
    perturbation = tf.sign(grad)
    adversarial = image + epsilon * perturbation
    adversarial = tf.clip_by_value(adversarial, clip_value_min=0.0, clip_value_max=1.0)
    return adversarial
