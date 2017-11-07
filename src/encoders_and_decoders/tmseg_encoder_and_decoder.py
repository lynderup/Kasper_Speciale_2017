import dataprovider.dataset_provider as dataset_provider
import dataprovider.mappings as mappings

observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

structure_encode_dict = {"1": mappings.INSIDE,
                         "H": mappings.HELIX,
                         "h": mappings.HELIX,
                         "2": mappings.OUTSIDE,
                         "U": mappings.UNKNOWN,
                         "0": mappings.UNKNOWN,
                         "L": mappings.UNKNOWN}

structure_decode_dict = {mappings.INSIDE: "i",
                         mappings.HELIX: "h",
                         mappings.OUTSIDE: "o",
                         mappings.UNKNOWN: "u"}

step1_target_decode_dict = {mappings.MEMBRANE: "M",
                            mappings.NOTMEMBRANE: "n"}

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

