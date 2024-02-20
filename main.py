import math
import time
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from qclassifier import Qclassifier

LOCAL_FOLDER = Path(__file__).parent


def main():
    # ==============
    # Configuration
    # ==============
    epochs = 2
    nclasses = 3
    training_size = 8 * nclasses
    validation_size = 2 * nclasses
    test_size = 2 * nclasses
    batch_size = 2
    resize = 8
    # layers = [1, 2, 3, 4, 5, 6]
    layers = [1, 2]
    seed = 1
    # block_sizes = [[2, 4], [3, 4], [4, 4], [4, 8], [8, 8]]
    block_sizes = [[resize, resize]]
    # nqubits = [8, 6, 4, 2, 1]
    nqubits = [1]
    pooling = "max"

    file_path = LOCAL_FOLDER / "statistics"

    for k in range(len(nqubits)):
        accuracy_qubits = []
        for j in range(len(layers)):
            # Create class
            my_class = Qclassifier(
                training_size=training_size,
                validation_size=validation_size,
                test_size=test_size,
                nepochs=epochs,
                batch_size=batch_size,
                nlayers=layers[j],
                seed_value=seed,
                nqubits=nqubits[k],
                resize=resize,
                nclasses=nclasses,
                pooling=pooling,
                block_width=block_sizes[k][0],
                block_height=block_sizes[k][1],
            )

            start_time = time.time()
            (
                trained_params,
                history_train_loss,
                history_val_loss,
                history_test_accuracy,
            ) = my_class.training_loop()
            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60

            if not os.path.exists("statistics"):
                os.makedirs("statistics")

            name_file = "trained_params_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            np.save(
                file_path / name_file,
                trained_params,
            )

            name_file = (
                "history_train_loss_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_train_loss,
            )

            name_file = (
                "history_val_loss_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_val_loss,
            )

            name_file = (
                "history_test_accuracy_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_test_accuracy,
            )


if __name__ == "__main__":
    main()
