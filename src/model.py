import tensorflow as tf

from dataprovider.tmseg_dataset_provider import TMSEGDatasetProvider


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
    num_output_classes = 4

    starting_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.96

    num_units = 10
    train_steps = 100

class Model:

    def __init__(self, dataprovider, config):
        self.dataprovider = dataprovider
        self.config = config

        self.train_step, self.iterator = self.build_model()

    def build_model(self):
        config = self.config

        dataset = self.dataprovider.get_dataset(config.batch_size)

        global_step = tf.Variable(0, trainable=False)

        iterator = dataset.make_initializable_iterator()
        lengths, sequences, structures = iterator.get_next()

        #  TODO: Tensors in wrong shapes. Need fixing!!!
        sequences = tf.transpose(sequences, perm=[1, 0])
        structures = tf.transpose(structures, perm=[1, 0])

        self.sequences = sequences

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

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss

        self.loss = loss

        train_step = optimizer.minimize(loss, global_step=global_step)

        return train_step, iterator



def train():
    dataprovider = TMSEGDatasetProvider()

    config = ModelConfig()
    m = Model(dataprovider, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(m.iterator.initializer)

        for i in range(config.train_steps):
            print(i)
            sess.run(m.train_step)


if __name__ == '__main__':
    train()