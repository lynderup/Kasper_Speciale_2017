import math

import tensorflow as tf
import numpy as np

import dataprovider.mappings as mappings
import dataprovider.dataset_provider as dataset_provider


def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.cast(sequence_lengths, tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


class ModelConfig:
    batch_size = 10
    num_input_classes = 20
    num_output_classes = 2

    starting_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.96

    num_units = 10
    train_steps = 100


class Model:
    def __init__(self, dataprovider, config, logdir):
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.config = config

        global_step = tf.Variable(0, trainable=False)

        # Data input
        with tf.variable_scope("Input", reuse=None):
            lengths, sequences, structures_step1, structures_step3 = self.build_data_input()
            self.lengths = lengths
            self.sequences = sequences
            self.structures_step1 = structures_step1
            self.structures_step3 = structures_step3

        # Build model graph
        with tf.variable_scope("Model", reuse=None):
            with tf.variable_scope("Step1", reuse=None):
                logits = self.build_model_step1(sequences, lengths)
                self.logits_step1 = logits

            with tf.variable_scope("Step3", reuse=None):
                self.embeddings_placeholder = tf.placeholder(self.embedded_input.dtype,
                                                             shape=self.embedded_input.get_shape(),
                                                             name="embeddings_placeholder")
                self.logits_placeholder = tf.placeholder(logits.dtype,
                                                         shape=logits.get_shape(),
                                                         name="logits_placeholder")
                self.targets_placeholder = tf.placeholder(tf.float32,  # structures_step3.dtype,
                                                          shape=self.structures_step3.get_shape(),
                                                          name="targets_placeholder")

                self.build_model_step3(embedding=self.embeddings_placeholder,
                                       logits=self.logits_placeholder,
                                       targets=self.targets_placeholder)

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

            with tf.variable_scope("Step3", reuse=None):
                var_list_step3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model/Step3')
                cross_entropy_loss_step3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_slices,
                                                                                   logits=self.sigmoid_output)
                cross_entropy_loss_step3 = tf.reduce_mean(cross_entropy_loss_step3)

                learning_rate, loss_step3, train_step_step3 = self.build_training_graph(cross_entropy_loss_step3,
                                                                                        var_list=var_list_step3,
                                                                                        global_step=global_step,
                                                                                        should_increment=False)
                self.loss_step3 = loss_step3
                self.train_step_step3 = train_step_step3

    def build_data_input(self):
        self.handle, self.iterator = self.dataprovider.get_iterator()
        lengths, sequences, structures_step1, structures_step3 = self.iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

        return lengths, sequences, structures_step1, structures_step3

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

    def build_model_step2(self, logits):
        pass

    def find_membranes(self, logits):
        batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
        batch_membranes = []

        for prediction in batch_predictions:

            in_membrane = False
            membranes = []
            membrane_start_index = -1

            for i, z in enumerate(prediction):
                if z == mappings.MEMBRANE:
                    if not in_membrane:
                        in_membrane = True
                        membrane_start_index = i
                else:
                    if in_membrane:
                        in_membrane = False
                        membranes.append((membrane_start_index, i))

            batch_membranes.append(np.asarray(membranes))

        return np.asarray(batch_membranes)

    def filter_membranes(self, batch_membranes):

        new_batch_membranes = []
        for membranes in batch_membranes:

            new_membranes = []
            for start, end in membranes:
                length = end - start

                if length > 5:
                    if length >= 35:
                        new_membranes.append((start, math.floor(length / 2)))
                        new_membranes.append((math.ceil(length / 2), end))
                    else:
                        new_membranes.append((start, end))
            new_batch_membranes.append(np.asarray(new_membranes))
        return np.asarray(new_batch_membranes)

    def numpy_step2(self, logits):
        batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
        batch_membranes = self.find_membranes(logits)

        new_predictions = []

        for i, membranes in enumerate(batch_membranes):
            prediction = batch_predictions[i]
            for start, end in membranes:
                length = end - start

                if length <= 5:
                    prediction[start:end] = [mappings.NOTMEMBRANE] * length

                if length >= 35:
                    new_membrane = [mappings.MEMBRANE] * length
                    new_membrane[math.floor(length / 2)] = mappings.NOTMEMBRANE
                    prediction[start:end] = new_membrane

            new_predictions.append(prediction)

        return np.asarray(new_predictions)

    def build_model_step3(self, embedding, logits, targets):
        config = self.config
        membrane_endpoints = tf.placeholder(tf.int32, shape=[None])
        batch_index = tf.placeholder(tf.int32, shape=[])

        self.membrane_endpoints = membrane_endpoints
        self.batch_index = batch_index

        # _input = tf.sigmoid(logits)
        _input = logits

        new_input = tf.concat([embedding, _input], axis=2)

        def new_input_slice_map(end_point):
            return tf.squeeze(tf.slice(new_input,
                                       [tf.maximum(0, tf.minimum(end_point - 5, tf.shape(new_input)[0] - 11)),
                                        batch_index, 0],
                                       [11, 1, -1]), axis=1)

        # Was once sequence major
        # def target_slice_map(end_point):
        #     return tf.squeeze(tf.slice(targets,
        #                                [tf.maximum(0, tf.minimum(end_point - 5, tf.shape(targets)[0] - 11)),
        #                                 batch_index],
        #                                [11, 1]), axis=1)

        def target_slice_map(end_point):
            return tf.squeeze(tf.slice(targets,
                                       [batch_index,
                                        tf.maximum(0, tf.minimum(end_point - 5, tf.shape(targets)[1] - 11))],
                                       [1, 11]), axis=0)

        input_slices = tf.map_fn(new_input_slice_map, membrane_endpoints, dtype=tf.float32)
        target_slices = tf.map_fn(target_slice_map, membrane_endpoints, dtype=tf.float32)

        self.input_slices = input_slices
        self.target_slices = target_slices

        window_size = 11
        input_size = 12
        num_hidden_units = 1000

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

    def build_training_graph(self, cross_entropy_loss, var_list, global_step, should_increment=False):
        config = self.config

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss

        if not should_increment:
            global_step = None

        train_step = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        return learning_rate, loss, train_step

    def train(self):
        summary_writer = tf.summary.FileWriter(self.logdir)
        summary_writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            train_handle, _ = sess.run(
                self.dataprovider.get_train_iterator_handle())  # get_handle returns (handle, init_op)
            train_feed = {handle: train_handle}

            validation_handle, _ = sess.run(
                self.dataprovider.get_validation_iterator_handle())  # get_handle returns (handle, init_op)
            validation_feed = {handle: validation_handle}

            sum_loss = tf.summary.scalar("loss", self.loss_step1)
            sum_val_loss = tf.summary.scalar("validation loss", self.loss_step1)
            sum_learn_rate = tf.summary.scalar("learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i)

                fetches = [merged_sum,
                           self.train_step_step1,
                           self.embedded_input,
                           self.structures_step3,
                           self.logits_step1]

                summary, _, emb, targets_step3, out_step1 = sess.run(fetches=fetches, feed_dict=train_feed)

                batch_membranes = self.find_membranes(out_step1)
                filtered_batch_membranes = self.filter_membranes(batch_membranes)

                for batch_index, membranes in enumerate(filtered_batch_membranes):
                    step3_feed = {self.membrane_endpoints: membranes.reshape(-1),  # Is list of pairs
                                  self.batch_index: batch_index,
                                  self.embeddings_placeholder: emb,
                                  self.logits_placeholder: out_step1,
                                  self.targets_placeholder: targets_step3,
                                  self.keep_prop: 0.5}

                    sess.run(self.train_step_step3, feed_dict=step3_feed)

                if i % 10 == 0:
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, i)
                    summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")

    def inference(self):

        lengths = self.lengths
        sequences = self.sequences
        structures_step1 = self.structures_step1
        structures_step3 = self.structures_step3
        logits = self.logits_step1

        embeddings = self.embedded_input

        with tf.Session() as sess:
            self.saver.restore(sess, self.logdir + "checkpoints/model.ckpt")

            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            test_handle, _ = sess.run(
                self.dataprovider.get_test_iterator_handle())  # get_handle returns (handle, init_op)
            test_feed = {handle: test_handle}

            fetches = [lengths, sequences, structures_step1, structures_step3, embeddings, logits]
            _lengths, inputs, targets_step1, targets_step3, emb, out = sess.run(fetches=fetches,
                                                                                feed_dict=test_feed)

            # Switch sequence dimension with batch dimension so it is batch-major
            batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            batch_inputs = np.swapaxes(inputs, 0, 1)
            batch_targets = np.swapaxes(targets_step1, 0, 1)

            batch_membranes = self.find_membranes(out)
            filtered_batch_membranes = self.filter_membranes(batch_membranes)
            filtered_batch_membranes_endpoints = np.asarray(
                [membranes.reshape(-1) for membranes in filtered_batch_membranes])

            step3_feed = {self.membrane_endpoints: filtered_batch_membranes_endpoints[0],
                          self.batch_index: 0,
                          self.embeddings_placeholder: emb,
                          self.logits_placeholder: out,
                          self.targets_placeholder: targets_step3,
                          self.keep_prop: 1}

            step3_logits, target_slices, loss = sess.run([self.sigmoid_output, self.target_slices, self.loss_step3],
                                                         feed_dict=step3_feed)
            # endpoint_corrections = np.argmax(step3_logits, axis=2)
            endpoint_corrections = step3_logits

            for i, (corrected, window) in enumerate(zip(endpoint_corrections, target_slices)):
                end_point = filtered_batch_membranes_endpoints[0][i]
                begin = np.maximum(0, np.minimum(end_point - 5, len(out) - 11))
                logit_slice = out[begin:begin + 11, 0, :]
                print(logit_slice.shape)

                print(np.argmax(window, axis=0))
                print(np.argmax(logit_slice, axis=1))
                print(np.argmax(corrected, axis=0))
                print()
            print(loss)

            batch_corrected_predictions = self.numpy_step2(out)

            predictions = zip(_lengths, batch_inputs, batch_targets, batch_predictions, batch_corrected_predictions)

        return predictions
