import numpy as np

import dataprovider.read_fasta as read_fasta

from dataprovider.dataset_provider import DatasetProvider

structure_dict = {"1": DatasetProvider.INSIDE,
                  "H": DatasetProvider.HELIX,
                  "h": DatasetProvider.HELIX,
                  "2": DatasetProvider.OUTSIDE,
                  "U": 0,
                  "0": 0,
                  "L": 0}

class TMSEGDatasetProvider(DatasetProvider):

    @classmethod
    def encode_structure(cls, structure):
        return np.asarray([structure_dict[z] for z in structure])

    @classmethod
    def decode_structure(cls, structure):
        pass

    @classmethod
    def fasta_entry_to_protein(cls, fasta_entry):
        entry_parts = fasta_entry.split("\n")

        sequence = cls.encode_sequence(entry_parts[1])
        structure = cls.encode_structure(entry_parts[2])

        p = cls.Protein(name=entry_parts[0], sequence=sequence, structure=structure)
        return p

    @classmethod
    def read_tmseg_dataset(cls):
        path = "datasets/tmseg/"
        set_path = "unmasked_hval0/"

        opm_set1 = "opm_set1.fasta"
        opm_set2 = "opm_set2.fasta"
        opm_set3 = "opm_set3.fasta"
        opm_set4 = "opm_set4.fasta"

        filename1 = "datasets/tmseg/opm_unmasked_hval0.fasta"
        filename2 = "datasets/tmseg/pdbtm_unmasked_hval0.fasta"

        filename = "".join([path, set_path, opm_set1])

        fasta_entries = read_fasta.read_fasta(filename)

        return [cls.fasta_entry_to_protein(entry) for entry in fasta_entries]


if __name__ == '__main__':
    # print(TMSEGDatasetProvider.encode_structure("HHH"))

    dataset = TMSEGDatasetProvider.read_tmseg_dataset()

    for entry in dataset:
        print(entry)
        # print(entry.sequence)
        # print(entry.structure)