import os

import tensorflow as tf
import numpy as np

import model.util as util


class ModelStep3:
    def __init__(self, config, logdir, dataprovider, handle, lengths, sequences, sequence_sup_data, target_step3):

        self.config = config
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.handle = handle

        self.lengths = lengths
        self.sequences = sequences
        self.sequence_sup_data = sequence_sup_data
        self.target_step3 = target_step3

        global_step = tf.Variable(0, trainable=False)

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            logits = self.build_model_step3(sequences, sequence_sup_data, lengths)
            self.logits_step3 = logits

        var_list_step3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Step3')
        print(var_list_step3)
        self.saver = tf.train.Saver(var_list_step3)

        # Build training graph step3
        with tf.variable_scope("Training", reuse=None):
            cross_entropy_loss_step3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_step3,
                                                                                      logits=logits)
            cross_entropy_loss_step3 = tf.reduce_mean(cross_entropy_loss_step3)

            learning_rate, loss_step3, train_step_step3 = self.build_training_graph(cross_entropy_loss_step3,
                                                                                    var_list=var_list_step3,
                                                                                    global_step=global_step)
            self.learning_rate = learning_rate
            self.loss_step3 = loss_step3
            self.train_step_step3 = train_step_step3

    def build_model_step3(self, sequences, sequence_sup_data, lengths):
        config = self.config

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units - 3], dtype=tf.float32)

        embedded_input = tf.nn.embedding_lookup(embedding, sequences)
        embedded_input = tf.concat([embedded_input, sequence_sup_data], axis=2)
        self.embedded_input = embedded_input

        bidirectional_output = util.add_bidirectional_lstm_layer(embedded_input,
                                                                 lengths,
                                                                 config.num_units,
                                                                 config.batch_size,
                                                                 sequence_output=False)

        output = util.add_lstm_layer(embedded_input, lengths, config.num_units , config.batch_size, sequence_output=False)

        logits = util.add_fully_connected_layer(output,
                                                config.num_units,
                                                config.num_output_classes,
                                                "softmax")

        return logits

    def build_training_graph(self, cross_entropy_loss, var_list, global_step):
        config = self.config

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        weights_list = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='Step3')
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

            handle = self.handle

            # get_handle returns (handle, init_op)
            train_handle, _ = sess.run(self.dataprovider.get_train_iterator_handle())
            train_feed = {handle: train_handle}

            # get_handle returns (handle, init_op)
            validation_handle, _ = sess.run(self.dataprovider.get_validation_iterator_handle())
            validation_feed = {handle: validation_handle}

            sum_loss = tf.summary.scalar("step3/loss", self.loss_step3)
            sum_val_loss = tf.summary.scalar("step3/validation loss", self.loss_step3)
            sum_learn_rate = tf.summary.scalar("step3/learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i, end=', ', flush=True)

                fetches = [merged_sum,
                           self.train_step_step3]

                summary, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                if i % 100 == 0:
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, i)
                    summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")