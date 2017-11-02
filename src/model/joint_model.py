import tensorflow as tf
import numpy as np
import model.util as util

import model.step1 as step1

class ModelConfig:
    batch_size = 10
    num_input_classes = 20
    num_output_classes = 2

    starting_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.96

    num_units = 10
    train_steps = 100


class Model:
    def __init__(self, dataprovider, config, logdir):
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.config = config

        global_step = tf.Variable(0, trainable=False)

        # Data input
        with tf.variable_scope("Input", reuse=None):
            handle, lengths, sequences, structures_step1, structures_step3 = self.build_data_input()
            self.lengths = lengths
            self.sequences = sequences
            self.structures_step1 = structures_step1
            self.structures_step3 = structures_step3

        self.model_step1 = step1.ModelStep1(config, logdir, dataprovider, handle, lengths, sequences, structures_step1)

    def build_data_input(self):
        handle, iterator = self.dataprovider.get_iterator()
        lengths, sequences, structures_step1, structures_step3 = iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

        return handle, lengths, sequences, structures_step1, structures_step3

    def train(self):
        self.model_step1.train()

    def inference(self):
        _lengths, inputs, targets_step1, out = self.model_step1.inference()

        # Switch sequence dimension with batch dimension so it is batch-major
        batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
        batch_inputs = np.swapaxes(inputs, 0, 1)
        batch_targets = np.swapaxes(targets_step1, 0, 1)

        batch_corrected_predictions = util.numpy_step2(out)

        predictions = zip(_lengths, batch_inputs, batch_targets, batch_predictions, batch_corrected_predictions)

        return predictions
