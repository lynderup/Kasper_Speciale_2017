
observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

INSIDE = 1
HELIX = 2
OUTSIDE = 3
UNKNOWN = 0

structure_encode_dict = {"1": INSIDE,
                         "H": HELIX,
                         "h": HELIX,
                         "2": OUTSIDE,
                         "U": UNKNOWN,
                         "0": UNKNOWN,
                         "L": UNKNOWN}

structure_decode_dict ={INSIDE: "i",
                        HELIX: "h",
                        OUTSIDE: "o",
                        UNKNOWN: "u"}


class TMSEGEncoder:

    def encode_sequence(self, sequence):
        return [observables.index(x) for x in sequence]

    def encode_structure(self, structure):
        return [structure_encode_dict[z] for z in structure]


class TMSEGDecoder:

    def decode_sequence(self, sequence):
        return "".join([observables[x] for x in sequence])

    def decode_structure(self, structure):
        return "".join([structure_decode_dict[z] for z in structure])

