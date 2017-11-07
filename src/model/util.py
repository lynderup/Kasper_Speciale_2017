import numpy as np
import tensorflow as tf
import math

import dataprovider.dataset_provider as dataset_provider


def find_membranes_aux(batch_predictions):
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


def find_membranes(logits):
    batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
    return find_membranes_aux(batch_predictions)


def filter_membranes(batch_membranes):

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


def numpy_step2(batch_predictions):
    batch_membranes = find_membranes_aux(batch_predictions)

    new_predictions = []

    for i, membranes in enumerate(batch_membranes):
        prediction = np.copy(batch_predictions[i])
        for start, end in membranes:
            length = end - start

            if length <= 5:
                prediction[start:end] = [dataset_provider.NOTMEMBRANE] * length

            if length >= 35:
                new_membrane = [dataset_provider.MEMBRANE] * length
                new_membrane[math.floor(length / 2)] = dataset_provider.NOTMEMBRANE
                prediction[start:end] = new_membrane

        new_predictions.append(prediction)

    return np.asarray(new_predictions)


def add_fully_connected_layer(input_tensor, input_size, output_size, name):

    weight = tf.get_variable(name + "_w", [input_size, output_size], dtype=tf.float32)
    bias = tf.get_variable(name + "_b", [output_size], dtype=tf.float32)

    logits = tf.matmul(input_tensor, weight) + bias

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    return logits


def add_bidirectional_lstm_layer(input_tensor, lengths, num_units, batch_size):
    fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units,
                                                forget_bias=0,
                                                cell_clip=None,
                                                use_peephole=False)
    bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)

    initial_state = (tf.zeros([batch_size, num_units], tf.float32),
                     tf.zeros([batch_size, num_units], tf.float32))

    fw_output, fw_state = fw_lstm(input_tensor,
                                  initial_state=initial_state,
                                  dtype=None,
                                  sequence_length=lengths,
                                  scope="fw_rnn")

    bw_output, bw_state = bw_lstm(input_tensor,
                                  initial_state=initial_state,
                                  dtype=None,
                                  sequence_length=lengths,
                                  scope="bw_rnn")

    with tf.variable_scope("fw_rnn", reuse=True):
        weight = tf.get_variable("kernel")
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    with tf.variable_scope("bw_rnn", reuse=True):
        weight = tf.get_variable("kernel")
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    output = tf.concat([fw_output, bw_output], axis=2)

    return output
