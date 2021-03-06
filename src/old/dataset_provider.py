import tensorflow as tf
import dataprovider.mappings as mappings


def scan_fn(acc, x):
    return tf.cond(tf.equal(x, mappings.MEMBRANE),
                   lambda: tf.cond(acc[1],
                                   lambda: (0, True),
                                   lambda: (1, True)),
                   lambda: (0, False))


structure_to_step1_target_table = mappings.dict_to_hashtable(mappings.structure_to_step1_target_dict)
# sequence_to_sup_data_dict_table = mappings.dict_to_hashtable(mappings.sequence_to_sup_data_dict)
sequence_to_sup_data_dict_table = mappings.dict_to_embedding_tensor(mappings.sequence_to_sup_data_dict)



def structure_to_step_targets(lengths, sequence, structure):
    # step1_target = tf.map_fn(lambda s: structure_to_step1_target_dict[s], structure)
    step1_target = structure_to_step1_target_table.lookup(structure)

    # sequence_sup_data = sequence_to_sup_data_dict_table.lookup(sequence)
    sequence_sup_data = tf.nn.embedding_lookup(sequence_to_sup_data_dict_table, sequence)

    initializer = (0, False)

    forward = tf.scan(scan_fn, step1_target, initializer=initializer, back_prop=False)[0]
    backward = tf.reverse(tf.scan(scan_fn,
                                  tf.reverse(step1_target, axis=[0]),
                                  initializer=initializer,
                                  back_prop=False)[0], axis=[0])

    # Assumption that no membrane has length one
    step3_target = forward + backward

    return lengths, sequence, sequence_sup_data, step1_target, step3_target


def _parse_function(example_proto):
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "structure": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=example_proto,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)
    lengths = tf.cast(context_parsed["length"], tf.int32)
    # sequence = tf.cast(sequence_parsed["sequence"], tf.int32)
    sequence = sequence_parsed["sequence"]
    # structure = tf.cast(sequence_parsed["structure"], tf.int32)
    structure = sequence_parsed["structure"]

    return lengths, sequence, structure


class DatasetProvider:
    def __init__(self, dataset_path, filenames, batch_size):
        self.dataset_path = dataset_path

        trainset, validationset, testset = filenames

        self.training_dataset = self.get_dataset(batch_size, trainset, repeat_shuffle=True)
        self.validation_dataset = self.get_dataset(batch_size, validationset, repeat_shuffle=True)
        self.test_dataset = self.get_dataset(batch_size, testset)

    def get_table_init_op(self):
        # init_op = tf.group(sequence_to_sup_data_dict_table.init,
        #                    structure_to_step1_target_table.init)
        return structure_to_step1_target_table.init

    def get_dataset(self, batch_size, filenames, repeat_shuffle=False):
        filename_suffix = ".tfrecord"
        paths = [self.dataset_path + filename + filename_suffix for filename in filenames]

        dataset = tf.contrib.data.TFRecordDataset(paths)
        dataset = dataset.map(_parse_function)
        dataset = dataset.map(structure_to_step_targets)
        if repeat_shuffle:
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None], [None, 3], [None], [None]))

        return dataset

    def get_iterator(self):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               self.training_dataset.output_types,
                                                               self.training_dataset.output_shapes)

        return handle, iterator

    def get_train_iterator_handle(self):
        # train_iterator = self.training_dataset.make_one_shot_iterator()
        train_iterator = self.training_dataset.make_initializable_iterator()
        return train_iterator.string_handle(), train_iterator.initializer

    def get_validation_iterator_handle(self):
        validation_iterator = self.validation_dataset.make_initializable_iterator()
        return validation_iterator.string_handle(), validation_iterator.initializer

    def get_test_iterator_handle(self):
        test_iterator = self.test_dataset.make_initializable_iterator()
        return test_iterator.string_handle(), test_iterator.initializer
