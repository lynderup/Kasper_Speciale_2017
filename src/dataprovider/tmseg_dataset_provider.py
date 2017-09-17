import os

from dataprovider.dataset_provider import DatasetProvider
from dataprovider.download_dataset import download_dataset
from dataprovider.fasta_to_tfrecord_converter import fasta_to_tfrecord


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


if __name__ == '__main__':
    TMSEGDatasetProvider.download_and_convert_if_not_existing()
