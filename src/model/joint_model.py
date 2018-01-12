from collections import namedtuple
import time
import os

import tensorflow as tf

import model.step1 as step1
import model.step3 as step3
import dataprovider.joint_dataprovider as joint_dataprovider

# import dataprovider.dataprovider_step1
# import dataprovider.dataprovider_step3

ModelConfig = namedtuple("ModelConfig", ["step1_config", "step3_config"])
StepConfig = namedtuple("StepConfig", ["batch_size",
                                       "num_input_classes",
                                       "num_output_classes",
                                       "starting_learning_rate",
                                       "decay_steps",
                                       "decay_rate",
                                       "num_units",
                                       "train_steps",
                                       "keep_prop"])

default_step1_config = StepConfig(batch_size=10,
                                  num_input_classes=20,
                                  num_output_classes=2,
                                  starting_learning_rate=0.01,
                                  decay_steps=10,
                                  decay_rate=0.99,
                                  num_units=50,  # 50
                                  train_steps=1000,
                                  keep_prop=1)

default_step3_config = StepConfig(batch_size=50,
                                  num_input_classes=20,
                                  num_output_classes=2,
                                  starting_learning_rate=0.01,
                                  decay_steps=10,
                                  decay_rate=0.99,
                                  num_units=50,  # 50
                                  train_steps=1000,
                                  keep_prop=1)

default_config = ModelConfig(step1_config=default_step1_config, step3_config=default_step3_config)


# class ModelConfig:
#     batch_size = 10
#     num_input_classes = 20
#     num_output_classes = 2
#
#     starting_learning_rate = 0.1
#     decay_steps = 100
#     decay_rate = 0.96
#
#     num_units = 13
#     train_steps = 1000


def name_from_config(step_config):
    name = "learn_" + str(step_config.starting_learning_rate) + \
           "_units_" + str(step_config.num_units) + \
           "_steps_" + str(step_config.train_steps)
    return name


def make_logdir(logdir, step_config):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    name = name_from_config(step_config)
    run_logdir = logdir + name + "_"

    run_number = 0
    suggested_run_name = run_logdir + str(run_number) + "/"

    while os.path.exists(suggested_run_name):
        run_number += 1
        suggested_run_name = run_logdir + str(run_number) + "/"

    os.mkdir(suggested_run_name)
    return suggested_run_name


class Model:
    def __init__(self, logdir, config=default_config, dataprovider=None, should_step1=True, should_step3=True):
        print("Building graph")
        start = time.time()

        tf.reset_default_graph()

        if dataprovider is None:
            dataprovider = joint_dataprovider.Dataprovider()

        if should_step1:
            self.step1_logdir = make_logdir(logdir + "step1/", config.step1_config)

            # Data input
            with tf.variable_scope("Input", reuse=None):
                dataprovider_step1 = dataprovider.get_step1_dataprovider(batch_size=config.step1_config.batch_size)
                step1_data = self.build_data_input_step1(dataprovider_step1)

            with tf.variable_scope("Step1", reuse=None):
                self.model_step1 = step1.ModelStep1(config.step1_config,
                                                    self.step1_logdir,
                                                    dataprovider_step1,
                                                    *step1_data)

        if should_step3:
            self.step3_logdir = make_logdir(logdir + "step3/", config.step3_config)

            # Data input
            with tf.variable_scope("Input", reuse=None):
                dataprovider_step3 = dataprovider.get_step3_dataprovider(batch_size=config.step3_config.batch_size)
                step3_data = self.build_data_input_step3(dataprovider_step3)

            with tf.variable_scope("Step3", reuse=None):
                self.model_step3 = step3.ModelStep3(config.step3_config,
                                                    self.step3_logdir,
                                                    dataprovider_step3,
                                                    *step3_data)

        print("Build graph in: %i milliseconds" % ((time.time() - start) * 1000))

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
        summary_writer = tf.summary.FileWriter(self.step1_logdir)
        summary_writer.add_graph(tf.get_default_graph())

        self.model_step1.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

    def train_step3(self):
        summary_writer = tf.summary.FileWriter(self.step3_logdir)
        summary_writer.add_graph(tf.get_default_graph())

        self.model_step3.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

    def inference(self):
        predictions = self.model_step1.inference()

        return predictions
