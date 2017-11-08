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


def flat_map(length, sequence, sequence_sup_data, step1_target):
    helix_list = find_helices(step1_target, length)

    dataset = tf.data.Dataset.from_tensor_slices(helix_list)
    dataset_positive = dataset.map(lambda helix: (helix[1],
                                                  tf.slice(sequence, [helix[0]], [helix[1]]),
                                                  tf.slice(sequence_sup_data, [helix[0], 0], [helix[1], 3]),
                                                  tf.slice(step1_target, [helix[0]], [helix[1]])))

    return dataset


class DataproviderStep3:
    def __init__(self, batch_size):
        self.dataset_path = "datasets/tmseg/data/sets/tfrecords/"

        trainset = ["opm_set1", "opm_set2"]
        validationset = ["opm_set3"]

        self.training_dataset = self.get_dataset(batch_size, trainset)
        self.validation_dataset = self.get_dataset(batch_size, validationset)

    # def get_table_init_op(self):
    #     return structure_to_step1_target_table.init

    def get_dataset(self, batch_size, filenames, repeat_shuffle=False):
        filename_suffix = ".tfrecord"
        paths = [self.dataset_path + filename + filename_suffix for filename in filenames]

        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(dataprovider_step1.parse_function)
        dataset = dataset.map(dataprovider_step1.structure_to_step_targets)
        dataset = dataset.flat_map(flat_map)
        # dataset = dataset.repeat(None)  # Infinite iterations
        # dataset = dataset.shuffle(buffer_size=1000)
        # dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None], [None, 3], [None]))

        return dataset

    def get_table_init_op(self):
        return dataprovider_step1.structure_to_step1_target_table.init

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
