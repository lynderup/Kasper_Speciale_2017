from collections import namedtuple
import time
import os

import tensorflow as tf
import numpy as np

import model.step1 as step1
import model.step3 as step3
import dataprovider.joint_dataprovider as joint_dataprovider
import model.util as util
import dataprovider.mappings as mappings

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
                                  train_steps=1000,
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


def build_data_input_step1(dataprovider):
    handle, iterator = dataprovider.get_iterator()
    lengths, sequences, sequence_sup_data, structures_step1 = iterator.get_next()

    #  TODO: Tensors in wrong shapes. Need fixing!!!
    sequence_sup_data = tf.transpose(sequence_sup_data, perm=[1, 0, 2])
    sequences = tf.transpose(sequences, perm=[1, 0])
    structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

    return handle, lengths, sequences, sequence_sup_data, structures_step1


def build_data_input_step3(dataprovider):
    handle, iterator = dataprovider.get_iterator()
    lengths, sequences, sequence_sup_data, targets_step3 = iterator.get_next()

    #  TODO: Tensors in wrong shapes. Need fixing!!!
    sequence_sup_data = tf.transpose(sequence_sup_data, perm=[1, 0, 2])
    sequences = tf.transpose(sequences, perm=[1, 0])

    return handle, lengths, sequences, sequence_sup_data, targets_step3


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
    def __init__(self, logdir, config=default_config, dataprovider=None):
        self.config = config
        self.logdir = logdir

        if dataprovider is None:
            self.dataprovider = joint_dataprovider.Dataprovider()
        else:
            self.dataprovider = dataprovider

    def build_test_data_input(self):
        # Test data input
        with tf.variable_scope("Input", reuse=None):
            test_dataprovider = self.dataprovider.get_test_dataprovider(batch_size=1)
            handle, lengths, sequences, sequence_sup_data, structures_step1 = \
                build_data_input_step1(test_dataprovider)

            step1_test_data = (handle, lengths, sequences, sequence_sup_data, structures_step1)
        return step1_test_data, test_dataprovider

    def build_step1(self, logdir, config=None, test_data=None):

        print("Building step1 graph")
        start = time.time()

        if config is None:
            config = self.config

        # Data input
        with tf.variable_scope("Input", reuse=None):
            if test_data is None:
                is_training = True
                step1_dataprovider = self.dataprovider.get_step1_dataprovider(batch_size=config.step1_config.batch_size)
                step1_data = build_data_input_step1(step1_dataprovider)
            else:
                is_training = False
                step1_dataprovider = None
                step1_data = test_data

        with tf.variable_scope("Step1", reuse=None):
            model_step1 = step1.ModelStep1(config.step1_config,
                                           logdir,
                                           is_training,
                                           step1_dataprovider,
                                           step1_data)

        print("Build step1 graph in: %i milliseconds" % ((time.time() - start) * 1000))
        return model_step1

    def build_step3(self, logdir, config=None, test_data=None):
        print("Building step3 graph")
        start = time.time()

        if config is None:
            config = self.config

        # Data input
        with tf.variable_scope("Input", reuse=None):
            if test_data is None:
                step3_dataprovider = self.dataprovider.get_step3_dataprovider(batch_size=config.step3_config.batch_size)
                step3_data = build_data_input_step3(step3_dataprovider)
                is_training = True
            else:
                step3_dataprovider = None
                step3_data = test_data
                is_training = False

        with tf.variable_scope("Step3", reuse=None):
            model_step3 = step3.ModelStep3(config.step3_config,
                                           logdir,
                                           is_training,
                                           step3_dataprovider,
                                           step3_data)

        print("Build step3 graph in: %i milliseconds" % ((time.time() - start) * 1000))
        return model_step3

    def train_step1(self, logdir=None):
        tf.reset_default_graph()

        if logdir is None:
            logdir = make_logdir(self.logdir + "step1/", self.config.step1_config)

        model_step1 = self.build_step1(logdir)

        summary_writer = tf.summary.FileWriter(logdir)
        summary_writer.add_graph(tf.get_default_graph())

        model_step1.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

        return logdir

    def train_step3(self, logdir=None):
        tf.reset_default_graph()

        if logdir is None:
            logdir = make_logdir(self.logdir + "step3/", self.config.step1_config)

        model_step3 = self.build_step3(logdir)

        summary_writer = tf.summary.FileWriter(logdir)
        summary_writer.add_graph(tf.get_default_graph())

        model_step3.train(summary_writer)

        summary_writer.flush()
        summary_writer.close()

        return logdir

    def inference(self, step1_logdir, step3_logdir=None):
        tf.reset_default_graph()

        test_step1_config = self.config.step1_config._replace(batch_size=1)
        test_step3_config = self.config.step3_config._replace(batch_size=1)
        config = ModelConfig(step1_config=test_step1_config, step3_config=test_step3_config)

        set_lengths = []
        set_inputs = []
        set_sup_data = []
        set_targets = []
        set_predictions = []

        start = time.time()
        step1_test_data, test_dataprovider = self.build_test_data_input()
        handle, lengths, sequences, sequence_sup_data, structures_step1 = step1_test_data

        model_step1 = self.build_step1(step1_logdir, config=config, test_data=step1_test_data)
        logits_step1 = model_step1.logits_step1

        with tf.Session() as sess:
            model_step1.restore(sess)

            sess.run(test_dataprovider.get_table_init_op())

            # get_handle returns (handle, init_op)
            test_handle, _ = sess.run(test_dataprovider.get_test_iterator_handle())
            test_feed = {handle: test_handle}

            fetches = [lengths, sequences, sequence_sup_data, structures_step1, logits_step1]

            try:
                while True:
                    _lengths, inputs, sup_data, targets_step1, out = sess.run(fetches=fetches, feed_dict=test_feed)

                    # Switch sequence dimension with batch dimension so it is batch-major
                    batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
                    batch_inputs = np.swapaxes(inputs, 0, 1)
                    batch_sup_data = np.swapaxes(sup_data, 0, 1)
                    batch_targets = np.swapaxes(targets_step1, 0, 1)

                    # batch_corrected_predictions = util.numpy_step2(out)

                    set_lengths.extend(_lengths)
                    set_inputs.extend(batch_inputs)
                    set_sup_data.extend(batch_sup_data)
                    set_targets.extend(batch_targets)
                    set_predictions.extend(batch_predictions)
            except tf.errors.OutOfRangeError:
                pass

        step1_time = time.time() - start

        start = time.time()
        corrected_predictions = util.numpy_step2(set_predictions)
        step2_time = time.time() - start

        start_time = time.time()
        if step3_logdir is not None:

            # Test data input
            with tf.variable_scope("Input", reuse=None):
                handle = None
                lengths = tf.placeholder(dtype=tf.int32, shape=[1], name="Lengths")
                sequences = tf.placeholder(dtype=tf.int32, shape=[1, None], name="Sequences")
                sequence_sup_data = tf.placeholder(dtype=tf.float32, shape=[1, None, 3], name="Sup_data")
                targets_step3 = None

                sequences_t = tf.transpose(sequences, perm=[1, 0])
                sequence_sup_data_t = tf.transpose(sequence_sup_data, perm=[1, 0, 2])

            step3_test_data = (handle, lengths, sequences_t, sequence_sup_data_t, targets_step3)

            model_step3 = self.build_step3(step3_logdir, config=config, test_data=step3_test_data)
            logits_step3 = model_step3.logits_step3

            with tf.Session() as sess:
                model_step3.restore(sess)

                new_predictions = []

                for length, sequence, sup_data, structure in zip(set_lengths, set_inputs, set_sup_data, corrected_predictions):
                    helices_segments = util.numpy_step3_preprocess(length, sequence, sup_data, structure)

                    structure = np.copy(structure)

                    for helix_segments in helices_segments:

                        start_before, end_before = helix_segments[0][0:2]

                        helix_probabilities = []
                        for start, end, sequence_segment, sup_data_segment, structure_segment in helix_segments:
                            segment_length = end - start

                            feed_dict = {lengths: [segment_length],
                                         sequences: [sequence_segment],
                                         sequence_sup_data: [sup_data_segment]}

                            fetches = logits_step3

                            out_step3 = sess.run(fetches=fetches, feed_dict=feed_dict)
                            helix_probability = out_step3[0][mappings.MEMBRANE]
                            helix_probabilities.append(np.asarray((helix_probability, start, end)))

                        helix_probabilities = np.asarray(helix_probabilities)
                        biggest_prob_index = np.argmax(helix_probabilities, axis=0)[0]
                        _, start, end = helix_probabilities[biggest_prob_index]
                        start = int(start)
                        end = int(end)

                        structure[start_before:end_before] = [mappings.NONMEMBRANE] * (end_before - start_before)
                        new_helix = [mappings.MEMBRANE] * (end - start)
                        structure[start:end] = new_helix

                    new_predictions.append(structure)

            predictions = (set_lengths,
                           set_inputs,
                           set_targets,
                           set_predictions,
                           corrected_predictions,
                           np.asarray(new_predictions))
        else:
            predictions = (set_lengths, set_inputs, set_targets, set_predictions, corrected_predictions, [])
        # predictions = zip(set_lengths, set_inputs, set_targets, set_predictions)

        step3_time = time.time() - start_time
        print("Inference time for step 1: %s" % step1_time)
        print("Inference time for step 2: %s" % step2_time)
        print("Inference time for step 3: %s" % step3_time)

        return predictions
