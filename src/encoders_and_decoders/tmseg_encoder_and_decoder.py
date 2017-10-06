
observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

INSIDE = 1
HELIX = 2
OUTSIDE = 3

structure_dict = {"1": INSIDE,
                  "H": HELIX,
                  "h": HELIX,
                  "2": OUTSIDE,
                  "U": 0,
                  "0": 0,
                  "L": 0}


class TMSEGEncoder:

    def encode_sequence(self, sequence):
        return [observables.index(x) for x in sequence]

    def encode_structure(self, structure):
        return [structure_dict[z] for z in structure]


class TMSEGDecoder:
    pass
