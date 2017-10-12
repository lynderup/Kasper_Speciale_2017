import dataprovider.dataset_provider as dataset_provider

observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

structure_encode_dict = {"1": dataset_provider.INSIDE,
                         "H": dataset_provider.HELIX,
                         "h": dataset_provider.HELIX,
                         "2": dataset_provider.OUTSIDE,
                         "U": dataset_provider.UNKNOWN,
                         "0": dataset_provider.UNKNOWN,
                         "L": dataset_provider.UNKNOWN}

structure_decode_dict = {dataset_provider.INSIDE: "i",
                         dataset_provider.HELIX: "h",
                         dataset_provider.OUTSIDE: "o",
                         dataset_provider.UNKNOWN: "u"}

step1_target_decode_dict = {dataset_provider.MEMBRANE: "M",
                            dataset_provider.NOTMEMBRANE: "n"}

class TMSEGEncoder:

    def encode_sequence(self, sequence):
        return [observables.index(x) for x in sequence]

    def encode_structure(self, structure):
        return [structure_encode_dict[z] for z in structure]


class TMSEGDecoder:

    def decode_sequence(self, sequence):
        return "".join([observables[x] for x in sequence])

    def decode_structure(self, structure):
        # return "".join([structure_decode_dict[z] for z in structure])
        return "".join([step1_target_decode_dict[z] for z in structure])

