import tensorflow as tf
import numpy as np

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
    train_steps = 1000


class Model:

    def __init__(self, dataprovider, config, logdir):
        self.logdir = logdir
        self.dataprovider = dataprovider
        self.config = config

        with tf.variable_scope("Model", reuse=None):
            self.train_step = self.build_model()

    def build_model(self):
        config = self.config

        # dataset = self.dataprovider.get_dataset(config.batch_size)
        # iterator = dataset.make_initializable_iterator()

        global_step = tf.Variable(0, trainable=False)

        self.handle, self.iterator = self.dataprovider.get_iterator()
        lengths, sequences, structures = self.iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures = tf.transpose(structures, perm=[1, 0])

        self.lengths = lengths
        self.sequences = sequences
        self.structures = structures

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

        self.logits = logits

        cross_entropy_loss = tf.reduce_mean(sequence_cross_entropy(labels=structures,
                                                                   logits=logits,
                                                                   sequence_lengths=lengths))

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate,
                                                   global_step,
                                                   config.decay_steps,
                                                   config.decay_rate,
                                                   staircase=True)

        self.learning_rate = learning_rate

        trainable_vars = tf.trainable_variables()

        self.saver = tf.train.Saver(trainable_vars)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss

        self.loss = loss

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
        structures = self.structures
        logits = self.logits

        with tf.Session() as sess:
            self.saver.restore(sess, self.logdir + "checkpoints/model.ckpt")

            sess.run(self.dataprovider.get_table_init_op())
            # sess.run(self.iterator.initializer)

            handle = self.handle
            test_handle, _ = sess.run(self.dataprovider.get_test_iterator_handle()) # get_handle returns (handle, init_op)
            test_feed = {handle: test_handle}

            len, inputs, targets, out = sess.run([lengths, sequences, structures, logits], feed_dict=test_feed)

            # Switch sequence dimension with batch dimension so it is batch-major
            batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            batch_inputs = np.swapaxes(inputs, 0, 1)
            batch_targets = np.swapaxes(targets, 0, 1)

            predictions = zip(len, batch_inputs, batch_targets, batch_predictions)

        return predictions
