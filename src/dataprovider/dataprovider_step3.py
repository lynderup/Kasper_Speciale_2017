import tensorflow as tf

import dataprovider.mappings as mappings
import dataprovider.dataprovider_step1 as dataprovider_step1


def find_helices(step1_target, length):
    def while_cond(i, helix_list, start_index, size, in_helix):
        return tf.less(i, length)

    # You just gotta love expression chaining
    def while_body(i, helix_list, start_index, size, in_helix):
        loop_vars = tf.cond(tf.equal(step1_target[i], mappings.MEMBRANE),
                            true_fn=lambda: (i + 1, helix_list, start_index, size + 1, True),
                            false_fn=lambda: tf.cond(in_helix,
                                                     true_fn=lambda: (i + 1,
                                                                      tf.concat([helix_list,
                                                                                 tf.expand_dims(tf.stack([start_index,
                                                                                                          size]),
                                                                                                axis=0)],
                                                                                axis=0),
                                                                      i + 1, 0, False),
                                                     false_fn=lambda: (i + 1, helix_list, i + 1, 0, False)))
        return loop_vars

    # index, (list of pairs of (start, size), index of start of current membrane,
    # length since start of current membrane, in membrane
    helix_list_init = tf.constant(0, shape=[0, 2], dtype=tf.int32)
    loop_vars_init = (tf.convert_to_tensor(0),
                      helix_list_init,
                      tf.constant(0),
                      tf.constant(0),
                      tf.constant(False))
    shape_invariants = (tf.TensorShape([]),
                        tf.TensorShape([None, 2]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([]))

    loop_vars = tf.while_loop(while_cond, while_body, loop_vars_init, shape_invariants=shape_invariants)
    _, helix_list, _, _, _ = loop_vars

    return helix_list


def flat_map(name, length, sequence, sequence_sup_data, pssm, step1_target):
    def lambda_from_offsets(start_offset, size_offset, target):
        return lambda helix: (name,
                              helix[1] + size_offset,
                              tf.slice(sequence,
                                       [helix[0] + start_offset],
                                       [helix[1] + size_offset]),
                              tf.slice(sequence_sup_data,
                                       [helix[0] + start_offset, 0],
                                       [helix[1] + size_offset, 3]),
                              tf.slice(pssm,
                                       [helix[0] + start_offset, 0],
                                       [helix[1] + size_offset, 20]),
                              target)
        # tf.slice(step1_target,
        #          [helix[0] + start_offset],
        #          [helix[1] + size_offset]))

    helix_list = find_helices(step1_target, length)

    dataset = tf.data.Dataset.from_tensor_slices(helix_list)
    no_shorter_than_6 = dataset.filter(lambda helix: tf.greater(helix[1], 6))
    starting_after_6 = dataset.filter(lambda helix: tf.greater(helix[0], 6))
    ending_before_6 = dataset.filter(lambda helix: tf.less(helix[0] + helix[1] + 6, length))

    dataset_positive = dataset.map(lambda_from_offsets(start_offset=0, size_offset=0, target=mappings.MEMBRANE))
    dataset_positive = dataset_positive.repeat(6)  # Compensating for more negative samples

    dataset_negative = no_shorter_than_6.map(
        lambda_from_offsets(start_offset=6, size_offset=-6, target=mappings.NONMEMBRANE))
    dataset_negative = dataset_negative.concatenate(no_shorter_than_6.map(
        lambda_from_offsets(start_offset=0, size_offset=-6, target=mappings.NONMEMBRANE)))

    dataset_negative = dataset_negative.concatenate(starting_after_6.map(
        lambda_from_offsets(start_offset=-6, size_offset=0, target=mappings.NONMEMBRANE)))
    dataset_negative = dataset_negative.concatenate(starting_after_6.map(
        lambda_from_offsets(start_offset=-6, size_offset=6, target=mappings.NONMEMBRANE)))

    dataset_negative = dataset_negative.concatenate(ending_before_6.map(
        lambda_from_offsets(start_offset=0, size_offset=6, target=mappings.NONMEMBRANE)))
    dataset_negative = dataset_negative.concatenate(ending_before_6.map(
        lambda_from_offsets(start_offset=6, size_offset=0, target=mappings.NONMEMBRANE)))

    dataset = dataset_positive.concatenate(dataset_negative)

    return dataset


class DataproviderStep3:
    def __init__(self, path):
        self.dataset_path = path

        self.dataprovider_step1 = dataprovider_step1.DataproviderStep1(path)

    def initilize_datasets(self, batch_size, trainset, validationset):
        self.training_dataset = self.get_dataset(batch_size, trainset, repeat_shuffle=True)
        self.validation_dataset = self.get_dataset(batch_size, validationset, repeat_shuffle=True)

    def get_dataset(self, batch_size, filenames, repeat_shuffle=False):
        dataset = self.dataprovider_step1.get_dataset(batch_size, filenames, repeat_shuffle=False, should_pad=False)
        dataset = dataset.flat_map(flat_map)
        if repeat_shuffle:
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([], [], [None], [None, 3], [None, 20], []))

        return dataset

    def get_table_init_op(self):
        return self.dataprovider_step1.get_table_init_op()

    def get_iterator(self):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               self.training_dataset.output_types,
                                                               self.training_dataset.output_shapes)

        return handle, iterator

    def get_train_iterator_handle(self):
        train_iterator = self.training_dataset.make_initializable_iterator()
        return train_iterator.string_handle(), train_iterator.initializer

    def get_validation_iterator_handle(self):
        validation_iterator = self.validation_dataset.make_initializable_iterator()
        return validation_iterator.string_handle(), validation_iterator.initializer
