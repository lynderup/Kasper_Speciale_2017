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
                                       "keep_prop",
                                       "l2_beta"])

default_step1_config = StepConfig(batch_size=10,
                                  num_input_classes=20,
                                  num_output_classes=2,
                                  starting_learning_rate=0.01,
                                  decay_steps=10,
                                  decay_rate=0.99,
                                  num_units=50,  # 50
                                  train_steps=1000,
                                  keep_prop=0.5,
                                  l2_beta=0.001)

default_step3_config = StepConfig(batch_size=50,
                                  num_input_classes=20,
                                  num_output_classes=2,
                                  starting_learning_rate=0.01,
                                  decay_steps=50,
                                  decay_rate=0.99,
                                  num_units=50,  # 50
                                  train_steps=10000,
                                  keep_prop=0.5,
                                  l2_beta=0.05)

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

        tf.reset_default_graph()
        self.config = config
        self.logdir = logdir

        self.step1_logdir = None
        self.step3_logdir = None
        self.model_step1 = None
        self.model_step3 = None

        if dataprovider is None:
            self.dataprovider = joint_dataprovider.Dataprovider()
        else:
            self.dataprovider = dataprovider

    def build_step1(self, logdir=None):
        if logdir is None:
            self.step1_logdir = make_logdir(self.logdir + "step1/", self.config.step1_config)
        else:
            self.step1_logdir = logdir

        print("Building step1 graph")
        start = time.time()

        with tf.variable_scope("Step1", reuse=None):
            self.model_step1 = step1.ModelStep1(self.config.step1_config,
                                                self.step1_logdir,
                                                self.dataprovider)

        print("Build step1 graph in: %i milliseconds" % ((time.time() - start) * 1000))

    def build_step3(self, logdir=None):
        if logdir is None:
            self.step3_logdir = make_logdir(self.logdir + "step3/", self.config.step3_config)
        else:
            self.step3_logdir = logdir

        print("Building step3 graph")
        start = time.time()

        with tf.variable_scope("Step3", reuse=None):
            self.model_step3 = step3.ModelStep3(self.config.step3_config,
                                                self.step3_logdir,
                                                self.dataprovider)

        print("Build step3 graph in: %i milliseconds" % ((time.time() - start) * 1000))

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
