import numpy as np
import tensorflow as tf
from datetime import datetime


def initialize_data(nclasses, training_size, test_size, validation_size, resize):
    """Method which prepares the validation, training and test datasets."""

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # ==============
    # Select classes of interest
    classes = list(range(nclasses))
    mask_train = np.isin(y_train, classes)
    mask_test = np.isin(y_test, classes)
    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # ==============
    # Perfect balance of the classes
    train_val_indices = []
    test_indices = []
    for class_label in classes:
        # Training
        train_indices_class = np.where(y_train == class_label)[0]
        sampled_indices_train = np.random.choice(
            train_indices_class,
            size=(training_size + validation_size) // nclasses,
            replace=False,
        )
        train_val_indices.extend(sampled_indices_train)

        # Test
        test_indices_class = np.where(y_test == class_label)[0]
        sampled_indices_test = np.random.choice(
            test_indices_class, size=test_size // nclasses, replace=False
        )
        test_indices.extend(sampled_indices_test)

    # TEST
    np.random.shuffle(test_indices)
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    # TRAINING AND VALIDATION
    # np.random.shuffle(train_val_indices)
    val_indices = []
    train_indices = []
    for n in classes:
        vindex_1 = (
            n * int(validation_size / nclasses) + int(training_size / nclasses) * n
        )
        vindex_2 = (
            int(validation_size / nclasses) * (n + 1)
            + int(training_size / nclasses) * n
        )
        tindex_1 = n * int(training_size / nclasses) + int(
            validation_size / nclasses
        ) * (n + 1)
        tindex_2 = (n + 1) * int(training_size / nclasses) + int(
            validation_size / nclasses
        ) * (n + 1)
        validation_pick = train_val_indices[vindex_1:vindex_2]
        train_pick = train_val_indices[tindex_1:tindex_2]

        val_indices.extend(validation_pick)
        train_indices.extend(train_pick)

    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    # ==============
    # Resizing
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)
    x_val = tf.expand_dims(x_val, axis=-1)

    x_train = tf.image.resize(x_train, [resize, resize])
    x_test = tf.image.resize(x_test, [resize, resize])
    x_val = tf.image.resize(x_val, [resize, resize])

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_val = x_val / 255.0

    training_data = (x_train, y_train)
    test_data = (x_test, y_test)
    validation_data = (x_val, y_val)

    return (
        training_data,
        test_data,
        validation_data,
    )


def pooling_creator(blocks, width, height, nqubits, mode):
    """Method to calculate the max or the average value of each block of an image.

    Args:
        blocks (float list): List where each element corresponds to the blocks into which we have divided the image.

    Returns:
        List of max values.
    """

    if mode == "max":
        max_values = []
        for i in range(nqubits):
            block = tf.reshape(blocks[i], [-1])
            max_values.append(tf.reduce_max(block))
        return max_values

    if mode == "average":
        average_values = []
        for i in range(nqubits):
            block = tf.reshape(blocks[i], [-1])
            mean = tf.reduce_sum(block) / (width * height)
            average_values.append(mean)
        return average_values


def block_creator(image, block_height, block_width):
    """Method to partition an image into blocks.

    Args:
        image: MNIST image.

    Returns:
        List containing images' blocks.
    """

    blocks = []
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            block = image[i : i + block_height, j : j + block_width]
            block = tf.reshape(block, [-1])
            blocks.append(block)
    return blocks


def shuffle(data):
    current_time = datetime.now()
    seconds_str = current_time.strftime("%S")
    seconds = int(seconds_str)

    x = tf.random.shuffle(data[0], seed=seconds)
    y = tf.random.shuffle(data[1], seed=seconds)

    return x, y
