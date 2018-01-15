import numpy as np
import tensorflow as tf
import math

import dataprovider.mappings as mappings


def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.cast(sequence_lengths, tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


def find_membranes_aux(batch_predictions):
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

            if length <= 7:
                prediction[start:end] = [mappings.NONMEMBRANE] * length

            if length >= 35:
                new_membrane = [mappings.MEMBRANE] * length
                new_membrane[math.floor(length / 2)] = mappings.NONMEMBRANE
                prediction[start:end] = new_membrane

        new_predictions.append(prediction)

    return np.asarray(new_predictions)


def numpy_step3_preprocess(sequence_length, sequence, sup_data, pssm, structure):
    sequence = sequence[0:sequence_length]
    sup_data = sup_data[0:sequence_length]
    pssm = pssm[0:sequence_length]
    structure = structure[0:sequence_length]

    helices = find_membranes_aux([structure])[0]
    helices_segments = []

    for start, end in helices:
        helix_segments = []
        length = end - start

        sequence_segment = sequence[start:end]
        sup_data_segment = sup_data[start:end]
        pssm_segment = pssm[start:end]
        structure_segment = structure[start:end]
        helix_segments.append((start, end, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

        if length > 6:
            sequence_segment = sequence[start + 6:end]
            sup_data_segment = sup_data[start + 6:end]
            pssm_segment = pssm[start + 6:end]
            structure_segment = structure[start + 6:end]
            helix_segments.append((start + 6, end, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

            sequence_segment = sequence[start:end - 6]
            sup_data_segment = sup_data[start:end - 6]
            pssm_segment = pssm[start:end - 6]
            structure_segment = structure[start:end - 6]
            helix_segments.append((start, end - 6, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

        if start > 6:
            sequence_segment = sequence[start - 6:end]
            sup_data_segment = sup_data[start - 6:end]
            pssm_segment = pssm[start - 6:end]
            structure_segment = structure[start - 6:end]
            helix_segments.append((start - 6, end, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

            sequence_segment = sequence[start - 6:end - 6]
            sup_data_segment = sup_data[start - 6:end - 6]
            pssm_segment = pssm[start - 6:end - 6]
            structure_segment = structure[start - 6:end - 6]
            helix_segments.append((start - 6, end - 6, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

        if end < sequence_length - 6:
            sequence_segment = sequence[start:end + 6]
            sup_data_segment = sup_data[start:end + 6]
            pssm_segment = pssm[start:end + 6]
            structure_segment = structure[start:end + 6]
            helix_segments.append((start, end + 6, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

            sequence_segment = sequence[start + 6:end + 6]
            sup_data_segment = sup_data[start + 6:end + 6]
            pssm_segment = pssm[start + 6:end + 6]
            structure_segment = structure[start + 6:end + 6]
            helix_segments.append((start + 6, end + 6, sequence_segment, sup_data_segment, pssm_segment, structure_segment))

        helices_segments.append(helix_segments)

    return helices_segments


def add_fully_connected_layer(input_tensor, input_size, output_size, name):
    weight = tf.get_variable(name + "_w", [input_size, output_size], dtype=tf.float32)
    bias = tf.get_variable(name + "_b", [output_size], dtype=tf.float32)

    logits = tf.matmul(input_tensor, weight) + bias

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    return logits


def add_bidirectional_lstm_layer(input_tensor, lengths, num_units, batch_size, sequence_output=True):
    fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units,
                                                forget_bias=0,
                                                cell_clip=None,
                                                use_peephole=False)
    bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)

    # Cell-state and previous output
    initial_state = (tf.zeros([batch_size, num_units], tf.float32),
                     tf.zeros([batch_size, num_units], tf.float32))

    fw_output, fw_state = fw_lstm(input_tensor,
                                  initial_state=initial_state,
                                  dtype=None,
                                  sequence_length=lengths,
                                  scope="fw_lstm")

    bw_output, bw_state = bw_lstm(input_tensor,
                                  initial_state=initial_state,
                                  dtype=None,
                                  sequence_length=lengths,
                                  scope="bw_lstm")

    with tf.variable_scope("fw_lstm", reuse=True):
        weight = tf.get_variable("kernel")
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    with tf.variable_scope("bw_lstm", reuse=True):
        weight = tf.get_variable("kernel")
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    if sequence_output:
        output = tf.concat([fw_output, bw_output], axis=2)
    else:
        output = tf.concat([fw_state.c, bw_state.c], axis=1)

    return output


def add_lstm_layer(input_tensor, lengths, num_units, batch_size, sequence_output=True):
    fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units,
                                                forget_bias=0,
                                                cell_clip=None,
                                                use_peephole=False)

    # Cell-state and previous output
    initial_state = (tf.zeros([batch_size, num_units], tf.float32),
                     tf.zeros([batch_size, num_units], tf.float32))

    output, state = fw_lstm(input_tensor,
                            initial_state=initial_state,
                            dtype=None,
                            sequence_length=lengths,
                            scope="lstm")

    with tf.variable_scope("lstm", reuse=True):
        weight = tf.get_variable("kernel")
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)

    if sequence_output:
        return output
    else:
        return state.c
