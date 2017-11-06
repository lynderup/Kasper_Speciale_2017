import os

import tensorflow as tf
import numpy as np

import model.util as util

class ModelStep3:

    def __init__(self,  config, logdir, dataprovider, handle, saver_step1, logdir_step1, embedded_input, logits_step1, structures_step3):

        self.config = config
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.handle = handle

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.saver_step1 = saver_step1
        self.logdir_step1 = logdir_step1

        self.embedded_input = embedded_input
        self.logits_step1 = logits_step1
        self.structures_step3 = structures_step3

        global_step = tf.Variable(0, trainable=False)

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            self.embeddings_placeholder = tf.placeholder(self.embedded_input.dtype,
                                                         shape=self.embedded_input.get_shape(),
                                                         name="embeddings_placeholder")
            self.logits_placeholder = tf.placeholder(self.logits_step1.dtype,
                                                     shape=self.logits_step1.get_shape(),
                                                     name="logits_placeholder")
            self.targets_placeholder = tf.placeholder(tf.float32,  # structures_step3.dtype,
                                                      shape=self.structures_step3.get_shape(),
                                                      name="targets_placeholder")

            self.build_model_step3(embedding=self.embeddings_placeholder,
                                   logits=self.logits_placeholder,
                                   targets=self.targets_placeholder)

        var_list_step3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Step3')
        self.saver = tf.train.Saver(var_list_step3)

        # Build training graph step1
        with tf.variable_scope("Training", reuse=None):

            learning_rate, loss_step3, train_step_step3 = self.build_training_graph(var_list=var_list_step3,
                                                                                    global_step=global_step)
            self.learning_rate = learning_rate
            self.loss_step3 = loss_step3
            self.train_step_step3 = train_step_step3


    def build_model_step3(self, embedding, logits, targets):
        # config = self.config

        window_radius = 15
        window_size = window_radius * 2 + 1
        input_size = 12
        num_hidden_units = 1000

        membrane_endpoints = tf.placeholder(tf.int32, shape=[None])
        batch_index = tf.placeholder(tf.int32, shape=[])

        self.membrane_endpoints = membrane_endpoints
        self.batch_index = batch_index

        # _input = tf.sigmoid(logits)
        _input = logits

        new_input = tf.concat([embedding, _input], axis=2)

        def new_input_slice_map(end_point):
            return tf.squeeze(tf.slice(new_input,
                                       [tf.maximum(0, tf.minimum(end_point - window_radius,
                                                                 tf.shape(new_input)[0] - window_size)),
                                        batch_index, 0],
                                       [window_size, 1, -1]), axis=1)

        def target_slice_map(end_point):
            return tf.squeeze(tf.slice(targets,
                                       [batch_index,
                                        tf.maximum(0, tf.minimum(end_point - window_radius,
                                                                 tf.shape(targets)[1] - window_size))],
                                       [1, window_size]), axis=0)

        input_slices = tf.map_fn(new_input_slice_map, membrane_endpoints, dtype=tf.float32, back_prop=False)
        target_slices = tf.map_fn(target_slice_map, membrane_endpoints, dtype=tf.float32, back_prop=False)

        self.input_slices = input_slices
        self.target_slices = target_slices

        hidden_w = tf.get_variable("hidden_w", [window_size * input_size, num_hidden_units], dtype=tf.float32)
        hidden_b = tf.get_variable("hidden_b", [num_hidden_units], dtype=tf.float32)

        self.keep_prop = tf.placeholder(tf.float32, shape=())

        _input_slices = tf.reshape(input_slices, shape=(-1, window_size * input_size))
        hidden_output = tf.sigmoid(tf.matmul(_input_slices, hidden_w) + hidden_b)
        # hidden_output = tf.nn.dropout(hidden_output, self.keep_prop)

        sigmoid_w = tf.get_variable("sigmoid_w", [num_hidden_units, window_size], dtype=tf.float32)
        sigmoid_b = tf.get_variable("sigmoid_b", [window_size], dtype=tf.float32)

        sigmoid_output = tf.matmul(hidden_output, sigmoid_w) + sigmoid_b
        # softmax_output = tf.reshape(_softmax_output, shape=(-1, window_size, config.num_output_classes))

        self.sigmoid_output = sigmoid_output

        self.l2_reg_loss = tf.nn.l2_loss(hidden_w) + tf.nn.l2_loss(sigmoid_w)

    def build_training_graph(self, var_list, global_step):
        config = self.config

        cross_entropy_loss_step3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_slices,
                                                                           logits=self.sigmoid_output)
        cross_entropy_loss_step3 = tf.reduce_mean(cross_entropy_loss_step3)

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss_step3 + self.l2_reg_loss

        train_step = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        return learning_rate, loss, train_step

    def train(self, summary_writer):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dataprovider.get_table_init_op())
            self.saver_step1.restore(sess, self.logdir_step1 + "checkpoints/model.ckpt")

            handle = self.handle
            train_handle, _ = sess.run(
                self.dataprovider.get_train_iterator_handle())  # get_handle returns (handle, init_op)
            train_feed = {handle: train_handle}

            validation_handle, _ = sess.run(
                self.dataprovider.get_validation_iterator_handle())  # get_handle returns (handle, init_op)
            validation_feed = {handle: validation_handle}

            sum_loss = tf.summary.scalar("step3/loss", self.loss_step3)
            sum_val_loss = tf.summary.scalar("step3/validation loss", self.loss_step3)
            sum_learn_rate = tf.summary.scalar("step3/learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i)

                fetches = [self.embedded_input,
                           self.structures_step3,
                           self.logits_step1]

                embeddings, targets_step3, out_step1 = sess.run(fetches=fetches, feed_dict=train_feed)

                batch_membranes = util.find_membranes(out_step1)
                filtered_batch_membranes = util.filter_membranes(batch_membranes)

                for batch_index, membranes in enumerate(filtered_batch_membranes):
                    step3_feed = {self.membrane_endpoints: membranes.reshape(-1),  # Is list of pairs
                                  self.batch_index: batch_index,
                                  self.embeddings_placeholder: embeddings,
                                  self.logits_placeholder: out_step1,
                                  self.targets_placeholder: targets_step3,
                                  self.keep_prop: 0.5}

                    summary, _ = sess.run([merged_sum, self.train_step_step3], feed_dict=step3_feed)

                    if batch_index == 0:
                        # val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                        # summary_writer.add_summary(val_loss, i)
                        summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")

    def inference(self):

        with tf.Session() as sess:
            self.saver_step1.restore(sess, self.logdir_step1 + "checkpoints/model.ckpt")
            self.saver.restore(sess, self.logdir + "checkpoints/model.ckpt")

            sess.run(self.dataprovider.get_table_init_op())

            handle = self.handle
            test_handle, _ = sess.run(
                self.dataprovider.get_test_iterator_handle())  # get_handle returns (handle, init_op)
            test_feed = {handle: test_handle}

            fetches = [self.embedded_input,
                       self.structures_step3,
                       self.logits_step1]

            embeddings, targets_step3, out_step1 = sess.run(fetches=fetches, feed_dict=test_feed)

            # Switch sequence dimension with batch dimension so it is batch-major
            # batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            # batch_inputs = np.swapaxes(inputs, 0, 1)
            # batch_targets = np.swapaxes(targets_step1, 0, 1)

            batch_membranes = util.find_membranes(out_step1)
            filtered_batch_membranes = util.filter_membranes(batch_membranes)
            filtered_batch_membranes_endpoints = np.asarray(
                [membranes.reshape(-1) for membranes in filtered_batch_membranes])

            print(filtered_batch_membranes_endpoints[0])

            step3_feed = {self.membrane_endpoints: filtered_batch_membranes_endpoints[0],
                          self.batch_index: 0,
                          self.embeddings_placeholder: embeddings,
                          self.logits_placeholder: out_step1,
                          self.targets_placeholder: targets_step3,
                          self.keep_prop: 1}

            fetches = [self.sigmoid_output, self.target_slices, self.loss_step3]

            step3_logits, target_slices, loss = sess.run(fetches=fetches, feed_dict=step3_feed)
            # endpoint_corrections = np.argmax(step3_logits, axis=2)
            endpoint_corrections = step3_logits

            for i, (corrected, window) in enumerate(zip(endpoint_corrections, target_slices)):
                end_point = filtered_batch_membranes_endpoints[0][i]
                begin = np.maximum(0, np.minimum(end_point - 5, len(out_step1) - 11))
                logit_slice = out_step1[begin:begin + 11, 0, :]
                print(logit_slice.shape)

                print(np.argmax(window, axis=0))
                print(np.argmax(logit_slice, axis=1))
                print(np.argmax(corrected, axis=0))
                print()
            print(loss)

            # batch_corrected_predictions = self.numpy_step2(out)
            # predictions = zip(_lengths, batch_inputs, batch_targets, batch_predictions, batch_corrected_predictions)

        # return predictions
