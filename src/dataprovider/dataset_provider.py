from collections import namedtuple

observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

class DatasetProvider:

    INSIDE = 1
    HELIX = 2
    OUTSIDE = 3

    Protein = namedtuple("Protein", ["name", "sequence", "structure"])

    @classmethod
    def encode_sequence(cls, sequence):
        return [observables.index(x) for x in sequence]

    @classmethod
    def decode_sequence(cls, sequence):
        return "".join([observables[x] for x in sequence])
