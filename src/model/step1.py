import os

import tensorflow as tf
import numpy as np

import model.util as util


def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.cast(sequence_lengths, tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


class ModelStep1:

    def __init__(self, config, logdir, dataprovider, handle, lengths, sequences, structures_step1):

        self.config = config
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.handle = handle

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.lengths = lengths
        self.sequences = sequences
        self.structures_step1 = structures_step1

        global_step = tf.Variable(0, trainable=False)

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            with tf.variable_scope("Step1", reuse=None):
                logits = self.build_model_step1(sequences, lengths)
                self.logits_step1 = logits

        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(trainable_vars)


        # Build training graph step1
        with tf.variable_scope("Training", reuse=None):
            with tf.variable_scope("Step1", reuse=None):
                var_list_step1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model/Step1')

                cross_entropy_loss_step1 = tf.reduce_mean(sequence_cross_entropy(labels=structures_step1,
                                                                                 logits=logits,
                                                                                 sequence_lengths=lengths))

                learning_rate, loss_step1, train_step_step1 = self.build_training_graph(cross_entropy_loss_step1,
                                                                                        var_list=var_list_step1,
                                                                                        global_step=global_step)
                self.learning_rate = learning_rate
                self.loss_step1 = loss_step1
                self.train_step_step1 = train_step_step1

    def build_model_step1(self, sequences, lengths):
        config = self.config

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units], dtype=tf.float32)

        embedded_input = tf.nn.embedding_lookup(embedding, sequences)
        self.embedded_input = embedded_input

        fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=config.num_units,
                                                    forget_bias=0,
                                                    cell_clip=None,
                                                    use_peephole=False)
        bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)

        initial_state = (tf.zeros([config.batch_size, config.num_units], tf.float32),
                         tf.zeros([config.batch_size, config.num_units], tf.float32))

        fw_output, fw_state = fw_lstm(embedded_input,
                                      initial_state=initial_state,
                                      dtype=None,
                                      sequence_length=lengths,
                                      scope="fw_rnn")

        bw_output, bw_state = bw_lstm(embedded_input,
                                      initial_state=initial_state,
                                      dtype=None,
                                      sequence_length=lengths,
                                      scope="bw_rnn")

        self.fw_output = fw_output

        softmax_w = tf.get_variable("softmax_w", [config.num_units * 2, config.num_output_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.num_output_classes], dtype=tf.float32)

        _fw_output = tf.reshape(fw_output, [-1, config.num_units])
        _bw_output = tf.reshape(bw_output, [-1, config.num_units])
        _output = tf.concat([_fw_output, _bw_output], 1)

        _logits = tf.matmul(_output, softmax_w) + softmax_b

        logits = tf.reshape(_logits, [-1, config.batch_size, config.num_output_classes])

        return logits

    def build_training_graph(self, cross_entropy_loss, var_list, global_step):
        config = self.config

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss

        train_step = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        return learning_rate, loss, train_step


    def train(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle

            # get_handle returns (handle, init_op)
            train_handle, _ = sess.run(self.dataprovider.get_train_iterator_handle())
            train_feed = {handle: train_handle}

            # get_handle returns (handle, init_op)
            validation_handle, _ = sess.run(self.dataprovider.get_validation_iterator_handle())
            validation_feed = {handle: validation_handle}

            sum_loss = tf.summary.scalar("loss", self.loss_step1)
            sum_val_loss = tf.summary.scalar("validation loss", self.loss_step1)
            sum_learn_rate = tf.summary.scalar("learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i)

                fetches = [merged_sum,
                           self.train_step_step1]

                summary, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                # if i % 10 == 0:
                #     val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                #     summary_writer.add_summary(val_loss, i)
                #     summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")

    def inference(self):

        lengths = self.lengths
        sequences = self.sequences
        structures_step1 = self.structures_step1
        logits = self.logits_step1

        with tf.Session() as sess:
            self.saver.restore(sess, self.logdir + "checkpoints/model.ckpt")

            sess.run(self.dataprovider.get_table_init_op())

            handle = self.handle

            # get_handle returns (handle, init_op)
            test_handle, _ = sess.run(self.dataprovider.get_test_iterator_handle())
            test_feed = {handle: test_handle}

            fetches = [lengths, sequences, structures_step1, logits]
            _lengths, inputs, targets_step1, out = sess.run(fetches=fetches,feed_dict=test_feed)

            # # Switch sequence dimension with batch dimension so it is batch-major
            batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            batch_inputs = np.swapaxes(inputs, 0, 1)
            batch_targets = np.swapaxes(targets_step1, 0, 1)

            batch_corrected_predictions = util.numpy_step2(out)

            predictions = zip(_lengths, batch_inputs, batch_targets, batch_predictions, batch_corrected_predictions)

        return predictions