import tensorflow as tf

import dataprovider.dataprovider_step1 as dataprovider_step1


class DataproviderTest:
    def __init__(self, path):
        self.dataset_path = path

        self.dataprovider_step1 = dataprovider_step1.DataproviderStep1(path)

    def initilize_datasets(self, batch_size, testset):
        self.test_dataset = self.dataprovider_step1.get_dataset(batch_size, testset, repeat_shuffle=False)

    def get_table_init_op(self):
        return self.dataprovider_step1.get_table_init_op()

    def get_iterator(self):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               self.test_dataset.output_types,
                                                               self.test_dataset.output_shapes)
        return handle, iterator

    def get_test_iterator_handle(self):
        test_iterator = self.test_dataset.make_initializable_iterator()
        return test_iterator.string_handle(), test_iterator.initializer
