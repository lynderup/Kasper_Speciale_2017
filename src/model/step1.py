import os

import tensorflow as tf
import numpy as np

import model.util as util


class ModelStep1:
    def __init__(self, config, logdir, dataprovider, handle, lengths, sequences, sequence_sup_data, structures_step1):

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
            logits = self.build_model_step1(sequences, sequence_sup_data, lengths)
            self.logits_step1 = logits

        var_list_step1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Step1')
        self.saver = tf.train.Saver(var_list_step1)

        # Build training graph step1
        with tf.variable_scope("Training", reuse=None):
            cross_entropy_loss_step1 = tf.reduce_mean(util.sequence_cross_entropy(labels=structures_step1,
                                                                                  logits=logits,
                                                                                  sequence_lengths=lengths))

            learning_rate, loss_step1, train_step_step1 = self.build_training_graph(cross_entropy_loss_step1,
                                                                                    var_list=var_list_step1,
                                                                                    global_step=global_step)
            self.learning_rate = learning_rate
            self.loss_step1 = loss_step1
            self.train_step_step1 = train_step_step1

    def build_model_step1(self, sequences, sequence_sup_data, lengths):
        config = self.config

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units - 3], dtype=tf.float32)

        embedded_input = tf.nn.embedding_lookup(embedding, sequences)
        embedded_input = tf.concat([embedded_input, sequence_sup_data], axis=2)
        self.embedded_input = embedded_input

        bidirectional_output = util.add_bidirectional_lstm_layer(embedded_input,
                                                                 lengths,
                                                                 config.num_units,
                                                                 config.batch_size)

        output = util.add_lstm_layer(bidirectional_output, lengths, config.num_units * 2, config.batch_size)
        # output = bidirectional_output

        _output = tf.reshape(output, [-1, config.num_units * 2])

        _logits = util.add_fully_connected_layer(_output, config.num_units * 2, config.num_output_classes, "softmax")

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

        weights_list = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='Step1')
        l2_reg_loss = 0
        for weight in weights_list:
            l2_reg_loss += tf.nn.l2_loss(weight)

        loss = cross_entropy_loss  # + l2_reg_loss

        train_step = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        return learning_rate, loss, train_step

    def train(self, summary_writer):

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

            sum_loss = tf.summary.scalar("step1/loss", self.loss_step1)
            sum_val_loss = tf.summary.scalar("step1/validation loss", self.loss_step1)
            sum_learn_rate = tf.summary.scalar("step1/learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i)

                fetches = [merged_sum,
                           self.train_step_step1]

                summary, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                if i % 10 == 0:
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, i)
                    summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")

    def inference(self):

        set_lengths = []
        set_inputs = []
        set_targets = []
        set_predictions = []

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

            for i in range(4):
                _lengths, inputs, targets_step1, out = sess.run(fetches=fetches, feed_dict=test_feed)

                # # Switch sequence dimension with batch dimension so it is batch-major
                batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
                batch_inputs = np.swapaxes(inputs, 0, 1)
                batch_targets = np.swapaxes(targets_step1, 0, 1)

                # batch_corrected_predictions = util.numpy_step2(out)

                set_lengths.extend(_lengths)
                set_inputs.extend(batch_inputs)
                set_targets.extend(batch_targets)
                set_predictions.extend(batch_predictions)

        # predictions = zip(set_lengths, set_inputs, set_targets, set_predictions)
        predictions = (set_lengths, set_inputs, set_targets, set_predictions)
        return predictions
