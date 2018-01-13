import os

import tensorflow as tf

import model.util as util


class ModelStep3:
    def __init__(self, config, logdir, is_training, dataprovider, step3_data):

        self.config = config
        self.logdir = logdir
        self.is_training = is_training
        self.dataprovider = dataprovider

        self.global_step = tf.Variable(0, trainable=False)

        handle, lengths, sequences, sequence_sup_data, targets_step3 = step3_data

        self.handle = handle

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            logits = self.build_model_step3(sequences, sequence_sup_data, lengths)
            self.logits_step3 = logits

        var_list_step3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Step3')
        self.saver = tf.train.Saver((*var_list_step3, self.global_step))

        if is_training:
            # Build training graph step3
            with tf.variable_scope("Training", reuse=None):
                cross_entropy_loss_step3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_step3,
                                                                                          logits=logits)
                cross_entropy_loss_step3 = tf.reduce_mean(cross_entropy_loss_step3)

                learning_rate, loss_step3, train_step_step3 = self.build_training_graph(cross_entropy_loss_step3,
                                                                                        var_list=var_list_step3,
                                                                                        global_step=self.global_step)
                self.learning_rate = learning_rate
                self.loss_step3 = loss_step3
                self.train_step_step3 = train_step_step3

    def build_model_step3(self, sequences, sequence_sup_data, lengths):
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
                                                                 config.batch_size,
                                                                 sequence_output=True)
        if self.is_training:
            bidirectional_output = tf.nn.dropout(bidirectional_output, keep_prop)

        output = util.add_lstm_layer(bidirectional_output,
                                     lengths,
                                     config.num_units * 2,
                                     config.batch_size, sequence_output=False)
        if self.is_training:
            output = tf.nn.dropout(output, keep_prop)

        logits = util.add_fully_connected_layer(output,
                                                config.num_units * 2,
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

            sum_loss = tf.summary.scalar("step3/loss", self.loss_step3)
            sum_val_loss = tf.summary.scalar("step3/validation loss", self.loss_step3)
            sum_learn_rate = tf.summary.scalar("step3/learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for _ in range(self.config.train_steps):
                fetches = [merged_sum,
                           global_step,
                           self.train_step_step3]

                summary, step, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                print(step, end=', ', flush=True)
                if step > 0 and step % 50 == 0:
                    print()

                if step % 10 == 0:
                    self.saver.save(sess, checkpoint_file, global_step=step)
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, step)
                    summary_writer.add_summary(summary, step)

            print()  # New line after steps counter
            self.saver.save(sess, checkpoint_file)

    def restore(self, sess):
        checkpoint_path = self.logdir + "checkpoints/"
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        self.saver.restore(sess, latest_checkpoint)