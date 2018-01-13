import os

import dataprovider.dataprovider_step1
import dataprovider.dataprovider_step3
import dataprovider.test_dataprovider

from dataprovider.utils.download_dataset import download_dataset
from dataprovider.utils.fasta_to_tfrecord_converter import fasta_to_tfrecord
from encoders.tmseg_encoder import TMSEGEncoder


class Dataprovider:
    def __init__(self, path=None, trainset=None, validationset=None, testset=None):

        if path is None:
            self.dataset_path = "datasets/tmseg/data/sets/tfrecords/"
            download_and_convert_if_not_existing()
        else:
            self.dataset_path = path

        if trainset is None:  # set all or none
            self.trainset = ["opm_set1", "opm_set2"]
            self.validationset = ["opm_set3"]
            self.testset = ["opm_set3"]
        else:
            self.trainset = trainset
            self.validationset = validationset
            self.testset = testset

    def get_step1_dataprovider(self, batch_size):
        step1_data = dataprovider.dataprovider_step1.DataproviderStep1(path=self.dataset_path)
        step1_data.initilize_datasets(batch_size=batch_size,
                                      trainset=self.trainset,
                                      validationset=self.validationset)
        return step1_data

    def get_step3_dataprovider(self, batch_size):
        step3_data = dataprovider.dataprovider_step3.DataproviderStep3(path=self.dataset_path)
        step3_data.initilize_datasets(batch_size=batch_size,
                                      trainset=self.trainset,
                                      validationset=self.validationset)
        return step3_data

    def get_test_dataprovider(self, batch_size):
        test_data = dataprovider.test_dataprovider.DataproviderTest(path=self.dataset_path)
        test_data.initilize_datasets(batch_size=batch_size,
                                     testset=self.testset)
        return test_data


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

    pdbtm_set1 = "pdbtm_set1"
    pdbtm_set2 = "pdbtm_set2"
    pdbtm_set3 = "pdbtm_set3"
    pdbtm_set4 = "pdbtm_set4"

    sets = [opm_set1, opm_set2, opm_set3, opm_set4, pdbtm_set1, pdbtm_set2, pdbtm_set3, pdbtm_set4]

    if not os.path.exists(path + tfrecord_path):
        print("Converting dataset")
        os.makedirs(path + tfrecord_path)
        encoder = TMSEGEncoder()
        fasta_to_tfrecord(path, fasta_path, tfrecord_path, sets, encoder)
