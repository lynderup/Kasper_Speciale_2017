import os
import tensorflow as tf

from dataprovider.dataset_provider import DatasetProvider
from dataprovider.download_dataset import download_dataset
from dataprovider.fasta_to_tfrecord_converter import fasta_to_tfrecord


def _parse_function(example_proto):
    context_features = {
        "length": tf.FixedLenFeature([1], dtype=tf.int64)
    }
    sequence_features = {
        "sequence": tf.FixedLenSequenceFeature([1], dtype=tf.int64),
        "structure": tf.FixedLenSequenceFeature([1], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=example_proto,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)

    return context_parsed["length"], sequence_parsed["sequence"], sequence_parsed["structure"]


class TMSEGDatasetProvider(DatasetProvider):

    @staticmethod
    def download_and_convert_if_not_existing():
        download_path = "https://github.com/Rostlab/TMSEG/raw/develop/supplementary/dataset-and-validation.tar.gz"
        save_path = "datasets/tmseg/"
        filename = "dataset-and-validation.tar.gz"

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            download_dataset(download_path, save_path, filename)

        path = "datasets/tmseg/data/sets/"
        fasta_path = "unmasked_hval0/"
        tfrecord_path = "tfrecords/"

        opm_set1 = "opm_set1"
        opm_set2 = "opm_set2"
        opm_set3 = "opm_set3"
        opm_set4 = "opm_set4"

        sets = [opm_set1, opm_set2, opm_set3, opm_set4]

        if not os.path.exists(path + tfrecord_path):
            print("Converting dataset")
            os.makedirs(path + tfrecord_path)
            fasta_to_tfrecord(path, fasta_path, tfrecord_path, sets)

    @staticmethod
    def get_dataset():
        dataset_path = "datasets/tmseg/data/sets/tfrecords/"
        filename_suffix = ".tfrecord"
        filenames = ["opm_set1", "opm_set2", "opm_set3"]

        paths = [dataset_path + filename + filename_suffix for filename in filenames]

        dataset = tf.contrib.data.TFRecordDataset(paths)
        dataset = dataset.map(_parse_function)

        iterator = dataset.make_initializable_iterator()
        example = iterator.get_next()
        print(example)

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            print(sess.run(example))


if __name__ == '__main__':
    TMSEGDatasetProvider.download_and_convert_if_not_existing()
    TMSEGDatasetProvider.get_dataset()
