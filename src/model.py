import math
import functools

import tensorflow as tf
import numpy as np

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

        #Data input
        with tf.variable_scope("Input", reuse=None):
            lengths, sequences, structures_step1 = self.build_data_input()
            self.lengths = lengths
            self.sequences = sequences
            self.structures_step1 = structures_step1


        # Build model graph step1
        with tf.variable_scope("Model", reuse=None):
            logits = self.build_model_step1(sequences, lengths)
            self.logits_step1 = logits

        # Build training graph step1
        with tf.variable_scope("Training", reuse=None):
            train_step = self.build_training_graph_step1(logits, structures_step1, lengths, global_step)
            self.train_step = train_step

    def build_data_input(self):
        self.handle, self.iterator = self.dataprovider.get_iterator()
        lengths, sequences, structures_step1 = self.iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures_step1 = tf.transpose(structures_step1, perm=[1, 0])

        return lengths, sequences, structures_step1

    def build_model_step1(self, sequences, lengths):
        config = self.config

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units], dtype=tf.float32)

        inputs = tf.nn.embedding_lookup(embedding, sequences)
        self.inputs = inputs

        fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=config.num_units,
                                                    forget_bias=0,
                                                    cell_clip=None,
                                                    use_peephole=False)
        bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)

        initial_state = (tf.zeros([config.batch_size, config.num_units], tf.float32),
                         tf.zeros([config.batch_size, config.num_units], tf.float32))

        fw_output, fw_state = fw_lstm(inputs,
                                      initial_state=initial_state,
                                      dtype=None,
                                      sequence_length=lengths,
                                      scope="fw_rnn")

        bw_output, bw_state = bw_lstm(inputs,
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

        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(trainable_vars)

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
                if z == dataset_provider.MEMBRANE:
                    if not in_membrane:
                        in_membrane = True
                        membrane_start_index = i
                else:
                    if in_membrane:
                        in_membrane = False
                        membranes.append((membrane_start_index, i))

            batch_membranes.append(np.asarray(membranes))

        return np.asarray(batch_membranes)

    def numpy_step2(self, logits):
        batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
        batch_membranes = self.find_membranes(logits)

        new_predictions = []

        for i, membranes in enumerate(batch_membranes):
            prediction = batch_predictions[i]
            for start, end in membranes:
                length = end - start

                if length <= 5:
                    prediction[start:end] = [dataset_provider.NOTMEMBRANE] * length

                if length >= 35:
                    new_membrane = [dataset_provider.MEMBRANE] * length
                    new_membrane[math.floor(length/2)] = dataset_provider.NOTMEMBRANE
                    prediction[start:end] = new_membrane

            new_predictions.append(prediction)

        return np.asarray(new_predictions)

    def build_model_step3(self, embedding, logits, targets):
        membrane_endpoints = tf.placeholder(tf.int32, shape=[None])
        batch_index = tf.placeholder(tf.int32, shape=[])

        self.membrane_endpoints = membrane_endpoints
        self.batch_index = batch_index

        new_input = tf.concat([embedding, logits], axis=2)

        def new_input_slice_map(end_point):
            return tf.squeeze(tf.slice(new_input,
                                       [tf.minimum(end_point - 5, tf.shape(new_input)[0] - 11), batch_index, 0],
                                       [11, 1, -1]), axis=1)

        def target_slice_map(end_point):
            return tf.squeeze(tf.slice(targets,
                                       [tf.minimum(end_point - 5, tf.shape(targets)[0] - 11), batch_index],
                                       [11, 1]), axis=1)

        input_slices = tf.map_fn(new_input_slice_map, membrane_endpoints, dtype=tf.int32)
        target_slices = tf.map_fn(target_slice_map, membrane_endpoints, dtype=tf.int64)

        self.input_slices = input_slices
        self.target_slices = target_slices

    def build_training_graph_step1(self, logits, targets, lengths, global_step):
        config = self.config

        cross_entropy_loss = tf.reduce_mean(sequence_cross_entropy(labels=targets,
                                                                   logits=logits,
                                                                   sequence_lengths=lengths))

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)
        self.learning_rate = learning_rate

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss
        self.loss = loss

        trainable_vars = tf.trainable_variables()
        train_step = optimizer.minimize(loss, var_list=trainable_vars, global_step=global_step)
        return train_step

    def train(self):
        summary_writer = tf.summary.FileWriter(self.logdir)
        summary_writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            train_handle, _ = sess.run(self.dataprovider.get_train_iterator_handle()) # get_handle returns (handle, init_op)
            train_feed = {handle: train_handle}

            validation_handle, _ = sess.run(self.dataprovider.get_validation_iterator_handle()) # get_handle returns (handle, init_op)
            validation_feed = {handle: validation_handle}

            sum_loss = tf.summary.scalar("loss", self.loss)
            sum_val_loss = tf.summary.scalar("validation loss", self.loss)
            sum_learn_rate = tf.summary.scalar("learning rate", self.learning_rate)

            merged_sum = tf.summary.merge([sum_loss, sum_learn_rate])

            for i in range(self.config.train_steps):
                print(i)

                fetches = [merged_sum, self.train_step]
                summary, _ = sess.run(fetches=fetches, feed_dict=train_feed)

                if i % 10 == 0:
                    val_loss = sess.run(sum_val_loss, feed_dict=validation_feed)
                    summary_writer.add_summary(val_loss, i)
                    summary_writer.add_summary(summary, i)

            self.saver.save(sess, self.logdir + "checkpoints/model.ckpt")

    def inference(self):

        lengths = self.lengths
        sequences = self.sequences
        structures = self.structures_step1
        logits = self.logits_step1

        embeddings = self.inputs

        embeddings_placeholder = tf.placeholder(embeddings.dtype, shape=embeddings.get_shape())
        logits_placeholder = tf.placeholder(logits.dtype, shape=logits.get_shape())
        targets_placeholder = tf.placeholder(structures.dtype, shape=structures.get_shape())
        self.build_model_step3(embedding=embeddings_placeholder, logits=logits_placeholder, targets=targets_placeholder)

        with tf.Session() as sess:
            self.saver.restore(sess, self.logdir + "checkpoints/model.ckpt")

            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            test_handle, _ = sess.run(self.dataprovider.get_test_iterator_handle()) # get_handle returns (handle, init_op)
            test_feed = {handle: test_handle}

            _lengths, inputs, targets, emb, out = sess.run([lengths, sequences, structures, embeddings, logits], feed_dict=test_feed)

            # Switch sequence dimension with batch dimension so it is batch-major
            batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            batch_inputs = np.swapaxes(inputs, 0, 1)
            batch_targets = np.swapaxes(targets, 0, 1)

            batch_membranes = self.find_membranes(out)
            print(batch_membranes[0])
            print(len(batch_membranes[0]))
            print(batch_membranes[0].reshape(-1))
            print(len(batch_membranes[0].reshape(-1)))

            step3_feed={self.membrane_endpoints:batch_membranes[0].reshape(-1),
                        self.batch_index:0,
                        embeddings_placeholder: emb,
                        logits_placeholder: out,
                        targets_placeholder: targets}

            slices = sess.run(self.target_slices, feed_dict=step3_feed)
            print(slices)
            print(len(slices))

            batch_corrected_predictions = self.numpy_step2(out)

            predictions = zip(_lengths, batch_inputs, batch_targets, batch_predictions, batch_corrected_predictions)

        return predictions
