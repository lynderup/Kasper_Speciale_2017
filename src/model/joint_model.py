import tensorflow as tf
import numpy as np
import model.util as util

import model.step1 as step1
import model.step3 as step3

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

        with tf.variable_scope("Step1", reuse=None):
            self.model_step1 = step1.ModelStep1(config,
                                                logdir + "step1/",
                                                dataprovider,
                                                handle,
                                                lengths,
                                                sequences,
                                                structures_step1)

        embedded_input = self.model_step1.embedded_input
        logits_step1 = self.model_step1.logits_step1

        with tf.variable_scope("Step3", reuse=None):
            self.model_step3 = step3.ModelStep3(config,
                                                logdir + "step3/",
                                                dataprovider,
                                                handle,
                                                self.model_step1.saver,
                                                logdir + "step1/",
                                                embedded_input,
                                                logits_step1,
                                                structures_step3)

        # var_list = tf.get_collection(tf.GraphKeys.WEIGHTS)
        # print(var_list)

    def build_data_input(self):
        handle, iterator = self.dataprovider.get_iterator()
        lengths, sequences, structures_step1, structures_step3 = iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

        return handle, lengths, sequences, structures_step1, structures_step3

    def train(self):
        summary_writer = tf.summary.FileWriter(self.logdir)
        summary_writer.add_graph(tf.get_default_graph())

        self.model_step1.train(summary_writer)

        # self.model_step3.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

    def inference(self):
        predictions = self.model_step1.inference()

        # self.model_step3.inference()

        return predictions
