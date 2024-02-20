import numpy as np
import tensorflow as tf
import math
from qibo.symbols import Z, I
from qibo import Circuit, gates, hamiltonians, set_backend
from data import initialize_data, pooling_creator, block_creator, shuffle
from help_functions import fidelity, create_target, number_params, create_target1


class Qclassifier:
    def __init__(
        self,
        training_size,
        validation_size,
        test_size,
        batch_size,
        nepochs,
        nlayers,
        pooling,
        seed_value,
        block_width,
        block_height,
        nqubits,
        nclasses,
        resize,
    ):

        np.random.seed(seed_value)
        set_backend("tensorflow")

        # TRAINING
        self.nclasses = nclasses
        self.training_size = training_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.learning_rate = [0.4, 0.2, 0.08, 0.04, 0.01, 0.005, 0.001]
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.targets = create_target1(nclasses)

        # IMAGE
        self.train, self.test, self.validation = initialize_data(
            nclasses,
            self.training_size,
            self.test_size,
            self.validation_size,
            resize,
        )
        self.block_width = block_width
        self.block_height = block_height

        # CIRCUIT
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.pooling = pooling
        self.n_embed_params = 2 * self.block_width * self.block_height * self.nqubits
        self.params_1layer = number_params(
            self.n_embed_params, self.nqubits, self.pooling
        )
        self.n_params = self.params_1layer * nlayers
        self.vparams = tf.Variable(tf.random.normal((self.n_params,)), dtype=tf.float32)
        self.hamiltonian = self.create_hamiltonian()
        self.ansatz = self.circuit()

    def create_hamiltonian(self):
        """Method for building the hamiltonian used to evaluate expectation values.

        Returns:
            qibo.hamiltonians.SymbolicHamiltonian()
        """
        ham = 0
        for k in range(self.nqubits):
            ham = I(0) * Z(k)
        hamiltonian = hamiltonians.SymbolicHamiltonian(ham)
        return hamiltonian

    def vparams_circuit(self, x):
        """Method which calculates the parameters necessary to embed an image in the circuit.

        Args:
            x: MNIST image.

        Returns:
            A list of parameters.
        """

        # Embedding angles
        blocks = block_creator(x, self.block_height, self.block_width)

        # Pooling angles
        pooling_values = pooling_creator(
            blocks, self.block_width, self.block_height, self.nqubits, self.pooling
        )

        angles = []
        for nlayer in range(self.nlayers):

            # Encoding params
            for j in range(self.nqubits):
                for i in range(self.block_width * self.block_height):
                    x = blocks[j][i]
                    angle = (
                        self.vparams[nlayer * self.params_1layer + i * 2] * x
                        + self.vparams[nlayer * self.params_1layer + (i * 2 + 1)]
                    )
                    angles.append(angle)

            # Pooling params
            if self.pooling != "no":
                for q in range(self.nqubits):
                    value = pooling_values[q]
                    angle = (
                        self.vparams[
                            nlayer * self.params_1layer + self.n_embed_params + 2 * q
                        ]
                        * value
                        + self.vparams[
                            nlayer * self.params_1layer
                            + self.n_embed_params
                            + (2 * q + 1)
                        ]
                    )
                    angles.append(angle)

        return angles

    def embedding_circuit(self):
        """Method for building the classifier's embedding block.

        Returns:
            Qibo circuit.
        """
        c = Circuit(self.nqubits)
        for j in range(self.nqubits):
            for i in range(self.block_width * self.block_height):
                if i % 3 == 1:
                    c.add(gates.RZ(j, theta=0))
                else:
                    c.add(gates.RY(j, theta=0))

        return c

    def pooling_circuit(self):
        """Method for building the classifier's pooling block.

        Returns:
            Qibo circuit.
        """
        c = Circuit(self.nqubits)
        for q in range(self.nqubits):
            c.add(gates.RX(q, theta=0))
        return c

    def entanglement_circuit(self):
        """Method for building the classifier's entanglement block.

        Returns:
            Qibo circuit.
        """

        c = Circuit(self.nqubits)
        for q in range(0, self.nqubits - 1, 2):
            c.add(gates.CNOT(q, q + 1))
        for q in range(1, self.nqubits - 2, 2):
            c.add(gates.CNOT(q, q + 1))
        c.add(gates.CNOT(self.nqubits - 1, 0))
        return c

    def circuit(self):
        """Method which builds the architecture layer by layer by summing Qibo circuits.

        Returns:
            qibo.Circuit()
        """
        circuit = Circuit(self.nqubits)

        for k in range(self.nlayers):
            # Embedding
            circuit += self.embedding_circuit()

            # Entanglement
            if self.nqubits != 1:
                circuit += self.entanglement_circuit()

            # Pooling
            if self.pooling != "no":
                circuit += self.pooling_circuit()

            # If last layer break the loop
            if k == self.nlayers - 1:
                break

            # Entanglement between layers
            if self.nqubits != 1:
                circuit += self.entanglement_circuit()

        with open("circuit.txt", "a") as file:
            print("=" * 60, file=file)
            print(circuit.draw(), file=file)
        return circuit

    def circuit_output(self, image):

        parameters = self.vparams_circuit(image)
        self.ansatz.set_parameters(parameters)

        # Execute the circuit
        result = self.ansatz()

        # Calculate the expectation value
        expectation_value = self.hamiltonian.expectation(result.state())

        return expectation_value, result.state()

    def accuracy(self):
        predictions_float, predicted_fids, _ = self.prediction_function(self.test)
        if self.nclasses == 2:
            accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            accuracy.update_state(self.test[1], predictions_float)
            accuracy = accuracy.result().numpy()
            return accuracy
        else:
            correct = sum(
                pred == label for pred, label in zip(predicted_fids, self.test[1])
            )
            total = len(self.test[1])
            accuracy = correct / total
            return accuracy

    def prediction_function(self, data):

        predictions_float = []
        predictions_fids = []
        predicted_states = []

        for x in data[0]:
            expectation, predicted_state = self.circuit_output(x)

            # Prediction float is a number between [0, nclasses-1]
            output = (self.nclasses - 1) * (expectation + 1) / 2

            # Prediction fid is the index corresponding to the highest fidelity
            # computed between the predicted state and the targets state
            fids = np.empty(len(self.targets))
            for j, y in enumerate(self.targets):
                fids[j] = fidelity(predicted_state, y)
            label = np.argmax(fids)

            # Append
            predictions_float.append(output)
            predictions_fids.append(label)
            predicted_states.append(predicted_state)

        return predictions_float, predictions_fids, predicted_states

    def loss_crossentropy(self, data, labels):
        """Loss function with two modes: fidelity loss and not_fidelity loss.

        Args:
            vparams: parameters of the architecture.
            batch_x: A set of training images.
            batch_y: The labels of the images.

        Returns:
            Loss function value of the current batch.
        """
        # Get predictions
        predictions_float, _, _ = self.prediction_function(data)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        cf = loss(labels, predictions)
        return cf

    def loss_fidelity(self, data, labels):
        cf = 0.0
        for i in range(self.batch_size):
            _, pred_state = self.circuit_output(data[i])
            label = tf.gather(labels, i)
            label = tf.cast(label, tf.int32)
            true_state = tf.gather(self.targets, label)
            cf += 0.5 * (1 - fidelity(pred_state, true_state)) ** 2

        return cf

    def train_step(self, x_batch, y_batch, optimizer):
        """Evaluate loss function on one train batch.

        Args:
            batch_size (int): number of samples in one training batch.
            encoder (qibo.models.Circuit): variational quantum circuit.
            params (tf.Variable): parameters of the circuit.
            vector (tf.Tensor): train sample, in the form of 1d vector.

        Returns:
            loss (tf.Variable): average loss of the training batch.
        """
        loss = 0.0

        if self.nclasses == 2:
            with tf.GradientTape() as tape:
                loss = self.loss_crossentropy(x_batch, y_batch)
            grads = tape.gradient(loss, self.vparams)
            grads = tf.math.real(grads)
            optimizer.apply_gradients(zip([grads], [self.vparams]))
            return loss

        if self.nclasses != 2:
            with tf.GradientTape() as tape:
                loss = self.loss_fidelity(x_batch, y_batch)
            grads = tape.gradient(loss, self.vparams)
            grads = tf.math.real(grads)
            optimizer.apply_gradients(zip([grads], [self.vparams]))
            return loss

    @tf.function
    def training_loop(self):
        """Method to train the classifier.

        Args:
            No

        Returns:
            No
        """
        trained_params = np.zeros((self.nepochs, self.n_params), dtype=float)
        history_train_loss = np.zeros((self.nepochs,), dtype=float)
        history_val_loss = np.zeros((self.nepochs,), dtype=float)
        history_test_accuracy = np.zeros((self.nepochs,), dtype=float)

        number_of_batches = math.ceil(self.training_size / self.batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        for epoch in range(self.nepochs):
            self.train = shuffle(self.train)
            for i in range(number_of_batches):
                loss = self.train_step(
                    self.train[0][i * self.batch_size : (i + 1) * self.batch_size],
                    self.train[1][i * self.batch_size : (i + 1) * self.batch_size],
                    optimizer,
                )
            trained_params[epoch] = self.vparams
            history_train_loss[epoch] = loss
            print("Epoch: %d  Loss: %f" % (epoch + 1, loss))

            if self.nclasses == 2:
                history_val_loss[epoch] = self.loss_crossentropy(
                    self.validation[0], self.validation[1]
                )
            else:
                history_val_loss[epoch] = self.loss_fidelity(
                    self.validation[0], self.validation[1]
                )

            history_test_accuracy[epoch] = self.accuracy()

        return (
            trained_params[-1],
            history_train_loss,
            history_val_loss,
            history_test_accuracy,
        )
