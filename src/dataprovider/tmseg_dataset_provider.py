import os
import tensorflow as tf

import dataprovider.dataset_provider as dataset_provider

from dataprovider.download_dataset import download_dataset
from dataprovider.fasta_to_tfrecord_converter import fasta_to_tfrecord
from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGEncoder


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


class TMSEGDatasetProvider(dataset_provider.DatasetProvider):

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
            encoder = TMSEGEncoder()
            fasta_to_tfrecord(path, fasta_path, tfrecord_path, sets, encoder)

    def get_dataset(self, batch_size):

        dataset_path = "datasets/tmseg/data/sets/tfrecords/"
        filename_suffix = ".tfrecord"
        filenames = ["opm_set1", "opm_set2", "opm_set3"]

        paths = [dataset_path + filename + filename_suffix for filename in filenames]

        dataset = tf.contrib.data.TFRecordDataset(paths)
        dataset = dataset.map(_parse_function)
        dataset = dataset.map(dataset_provider.structure_to_step_targets)
        dataset = dataset.repeat(None)  # Infinite iterations
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None], [None]))

        return dataset


if __name__ == '__main__':
    TMSEGDatasetProvider.download_and_convert_if_not_existing()

    dataprovider = TMSEGDatasetProvider()

    dataset = dataprovider.get_dataset(4)

    iterator = dataset.make_initializable_iterator()
    length, sequence, structure = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        print(sess.run(length))
        print(sess.run(sequence))
        print(sess.run(structure))
