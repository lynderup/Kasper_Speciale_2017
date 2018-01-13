import os

import tensorflow as tf
import numpy as np

import model.util as util


def build_data_input_step1(dataprovider):
    handle, iterator = dataprovider.get_iterator()
    lengths, sequences, sequence_sup_data, structures_step1 = iterator.get_next()

    #  TODO: Tensors in wrong shapes. Need fixing!!!
    sequence_sup_data = tf.transpose(sequence_sup_data, perm=[1, 0, 2])
    sequences = tf.transpose(sequences, perm=[1, 0])
    structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

    return handle, lengths, sequences, sequence_sup_data, structures_step1


class ModelStep1:
    def __init__(self, config, logdir, dataprovider):

        self.config = config
        self.logdir = logdir

        self.global_step = tf.Variable(0, trainable=False)

        # Data input
        with tf.variable_scope("Input", reuse=None):
            self.dataprovider = dataprovider.get_step1_dataprovider(batch_size=self.config.batch_size)
            handle, lengths, sequences, sequence_sup_data, structures_step1 = build_data_input_step1(self.dataprovider)

            self.handle = handle
            self.lengths = lengths
            self.sequences = sequences
            self.structures_step1 = structures_step1

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            logits = self.build_model_step1(sequences, sequence_sup_data, lengths)
            self.logits_step1 = logits

        var_list_step1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Step1')
        self.saver = tf.train.Saver((*var_list_step1, self.global_step))

        # Build training graph step1
        with tf.variable_scope("Training", reuse=None):
            cross_entropy_loss_step1 = tf.reduce_mean(util.sequence_cross_entropy(labels=structures_step1,
                                                                                  logits=logits,
                                                                                  sequence_lengths=lengths))

            learning_rate, loss_step1, train_step_step1 = self.build_training_graph(cross_entropy_loss_step1,
                                                                                    var_list=var_list_step1,
                                                                                    global_step=self.global_step)
            self.learning_rate = learning_rate
            self.loss_step1 = loss_step1
            self.train_step_step1 = train_step_step1

    def build_model_step1(self, sequences, sequence_sup_data, lengths):
        config = self.config

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units - 3], dtype=tf.float32)

        embedded_input = tf.nn.embedding_lookup(embedding, sequences)
        embedded_input = tf.concat([embedded_input, sequence_sup_data], axis=2)
        self.embedded_input = embedded_input

        keep_prop = tf.Variable(1, trainable=False, dtype=tf.float32, name="keep_prop")
        self.keep_prop = keep_prop

        bidirectional_output = util.add_bidirectional_lstm_layer(embedded_input,
                                                                 lengths,
                                                                 config.num_units,
                                                                 config.batch_size)
        bidirectional_output = tf.nn.dropout(bidirectional_output, keep_prop)

        output = util.add_lstm_layer(bidirectional_output, lengths, config.num_units * 2, config.batch_size)
        output = tf.nn.dropout(output, keep_prop)

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

        loss = cross_entropy_loss + config.l2_beta * l2_reg_loss

        train_step = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        return learning_rate, loss, train_step

    def train(self, summary_writer):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            checkpoint_path = self.logdir + "checkpoints/"
            checkpoint_file = checkpoint_path + "model.ckpt"

            if os.path.exists(checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
                self.saver.restore(sess, latest_checkpoint)

            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            keep_prop = self.keep_prop
            global_step = self.global_step

            # get_handle returns (handle, init_op)
            train_handle, _ = sess.run(self.dataprovider.get_train_iterator_handle())
            train_feed = {handle: train_handle,
                          keep_prop: self.config.keep_prop}

            # get_handle returns (handle, init_op)
            validation_handle, _ = sess.run(self.dataprovider.get_validation_iterator_handle())
            validation_feed = {handle: validation_handle,
                               keep_prop: 1.0}

            sum_loss = tf.summary.scalar("step1/loss", self.loss_step1)
            sum_val_loss = tf.summary.scalar("step1/validation loss", self.loss_step1)
            sum_learn_rate = tf.summary.scalar("step1/learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for _ in range(self.config.train_steps):

                fetches = [merged_sum,
                           global_step,
                           self.train_step_step1]

                summary, step, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                print(step, end=', ', flush=True)
                if step > 0 and step % 50 == 0:
                    print()

                if step % 10 == 0:
                    self.saver.save(sess, checkpoint_file, global_step=step)
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, step)
                    summary_writer.add_summary(summary, step)

            self.saver.save(sess, checkpoint_file)

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
            checkpoint_path = self.logdir + "checkpoints/"
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            self.saver.restore(sess, latest_checkpoint)

            sess.run(self.dataprovider.get_table_init_op())

            handle = self.handle
            keep_prop = self.keep_prop

            # get_handle returns (handle, init_op)
            test_handle, _ = sess.run(self.dataprovider.get_test_iterator_handle())
            test_feed = {handle: test_handle,
                         keep_prop: 1.0}

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
