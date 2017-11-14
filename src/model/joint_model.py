import tensorflow as tf
import time

import model.step1 as step1
import model.step3 as step3
import dataprovider.dataprovider_step1
import dataprovider.dataprovider_step3


class ModelConfig:
    batch_size = 10
    num_input_classes = 20
    num_output_classes = 2

    starting_learning_rate = 0.1
    decay_steps = 100
    decay_rate = 0.96

    num_units = 13
    train_steps = 1000


class Model:
    def __init__(self, config, logdir):
        print("Building graph")
        start = time.time()

        self.logdir = logdir
        self.config = config

        global_step = tf.Variable(0, trainable=False)

        # Data input
        with tf.variable_scope("Input", reuse=None):
            dataprovider_step1 = dataprovider.dataprovider_step1.DataproviderStep1(batch_size=config.batch_size)
            step1_data = self.build_data_input_step1(dataprovider_step1)

            dataprovider_step3 = dataprovider.dataprovider_step3.DataproviderStep3(batch_size=config.batch_size)
            step3_data = self.build_data_input_step3(dataprovider_step3)

        with tf.variable_scope("Step1", reuse=None):
            self.model_step1 = step1.ModelStep1(config,
                                                logdir + "step1/",
                                                dataprovider_step1,
                                                *step1_data)

        with tf.variable_scope("Step3", reuse=None):
            self.model_step3 = step3.ModelStep3(config,
                                                logdir + "step3/",
                                                dataprovider_step3,
                                                *step3_data)

        embedded_input = self.model_step1.embedded_input
        logits_step1 = self.model_step1.logits_step1

        print("Build graph in: %i milliseconds" % ((time.time() - start) * 1000))

        # var_list = tf.get_collection(tf.GraphKeys.WEIGHTS)
        # print(var_list)

    def build_data_input_step1(self, dataprovider):
        handle, iterator = dataprovider.get_iterator()
        lengths, sequences, sequence_sup_data, structures_step1 = iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequence_sup_data = tf.transpose(sequence_sup_data, perm=[1, 0, 2])
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

        return handle, lengths, sequences, sequence_sup_data, structures_step1

    def build_data_input_step3(self, dataprovider):
        handle, iterator = dataprovider.get_iterator()
        lengths, sequences, sequence_sup_data, targets_step3 = iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequence_sup_data = tf.transpose(sequence_sup_data, perm=[1, 0, 2])
        sequences = tf.transpose(sequences, perm=[1, 0])

        return handle, lengths, sequences, sequence_sup_data, targets_step3

    def train(self):
        summary_writer = tf.summary.FileWriter(self.logdir)
        summary_writer.add_graph(tf.get_default_graph())

        # self.model_step1.train(summary_writer)
        self.model_step3.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

    def inference(self):
        predictions = self.model_step1.inference()

        return predictions
