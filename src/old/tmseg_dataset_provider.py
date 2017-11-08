import os

import tensorflow as tf

import old.dataset_provider as dataset_provider
from dataprovider.utils.download_dataset import download_dataset
from dataprovider.utils.fasta_to_tfrecord_converter import fasta_to_tfrecord
from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGEncoder


class TMSEGDatasetProvider(dataset_provider.DatasetProvider):

    def __init__(self, batch_size):

        dataset_path = "datasets/tmseg/data/sets/tfrecords/"
        # filenames = ["opm_set1", "opm_set2", "opm_set3"]

        trainset = ["opm_set1", "opm_set2"]
        validationset = ["opm_set3"]
        # testset = ["opm_set4"]
        testset = ["opm_set3"] # To not overfit hyperparameters on testset

        filenames = (trainset, validationset, testset)

        super().__init__(dataset_path, filenames, batch_size)


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
