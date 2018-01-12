import tensorflow as tf
import dataprovider.mappings as mappings


def parse_function(example_proto):
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
    sequence = sequence_parsed["sequence"]
    structure = sequence_parsed["structure"]

    return lengths, sequence, structure


class DataproviderStep1:
    def __init__(self, path):
        self.structure_to_step1_target_table = mappings.dict_to_hashtable(mappings.structure_to_step1_target_dict)
        self.sequence_to_sup_data_dict_table = mappings.dict_to_embedding_tensor(mappings.sequence_to_sup_data_dict)

        self.dataset_path = path

    def initilize_datasets(self, batch_size, trainset, validationset, testset):
        self.training_dataset = self.get_dataset(batch_size, trainset, repeat_shuffle=True)
        self.validation_dataset = self.get_dataset(batch_size, validationset, repeat_shuffle=True)
        self.test_dataset = self.get_dataset(batch_size, testset)

    def get_table_init_op(self):
        return self.structure_to_step1_target_table.init

    def structure_to_step_targets(self, lengths, sequence, structure):
        step1_target = self.structure_to_step1_target_table.lookup(structure)

        sequence_sup_data = tf.nn.embedding_lookup(self.sequence_to_sup_data_dict_table, sequence)

        return lengths, sequence, sequence_sup_data, step1_target

    def get_dataset(self, batch_size, filenames, repeat_shuffle=False, should_pad=True):
        filename_suffix = ".tfrecord"
        paths = [self.dataset_path + filename + filename_suffix for filename in filenames]

        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(parse_function)
        dataset = dataset.map(self.structure_to_step_targets)
        if repeat_shuffle:
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=1000)
        if should_pad:
            dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None], [None, 3], [None]))

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
