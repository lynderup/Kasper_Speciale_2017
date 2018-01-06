import dataprovider.dataprovider_step1
import dataprovider.dataprovider_step3


class Dataprovider:

    def __init__(self, path=None, trainset=None, validationset=None, testset=None):

        if path is None:
            self.dataset_path = "datasets/tmseg/data/sets/tfrecords/"
        else:
            self.dataset_path = path

        if trainset is None: # set all or none
            self.trainset = ["opm_set1", "opm_set2"]
            self.validationset = ["opm_set3"]
            self.testset = ["opm_set3"]
        else:
            self.trainset = trainset
            self.validationset = validationset
            self.testset = testset

    def get_step1_dataprovider(self, batch_size):
        return dataprovider.dataprovider_step1.DataproviderStep1(batch_size=batch_size,
                                                                 path=self.dataset_path,
                                                                 trainset=self.trainset,
                                                                 validationset=self.validationset,
                                                                 testset=self.testset)

    def get_step3_dataprovider(self, batch_size):
        return dataprovider.dataprovider_step3.DataproviderStep3(batch_size=batch_size,
                                                                 path=self.dataset_path,
                                                                 trainset=self.trainset,
                                                                 validationset=self.validationset)