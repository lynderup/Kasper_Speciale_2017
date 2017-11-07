import dataprovider.dataset_provider as dataset_provider
import dataprovider.mappings as mappings

observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

sequence_encode_dict = {"A": mappings.ALA,
                        "R": mappings.ARG,
                        "N": mappings.ASN,
                        "D": mappings.ASP,
                        "C": mappings.CYS,
                        "E": mappings.GLU,
                        "Q": mappings.GLN,
                        "G": mappings.GLY,
                        "H": mappings.HIS,
                        "I": mappings.ILE,
                        "L": mappings.LEU,
                        "K": mappings.LYS,
                        "M": mappings.MET,
                        "F": mappings.PHE,
                        "P": mappings.PRO,
                        "S": mappings.SER,
                        "T": mappings.THR,
                        "W": mappings.TRP,
                        "Y": mappings.TYR,
                        "V": mappings.VAL}

structure_encode_dict = {"1": mappings.INSIDE,
                         "H": mappings.HELIX,
                         "h": mappings.HELIX,
                         "2": mappings.OUTSIDE,
                         "U": mappings.UNKNOWN,
                         "0": mappings.UNKNOWN,
                         "L": mappings.LOOP}

structure_decode_dict = {mappings.INSIDE: "i",
                         mappings.HELIX: "h",
                         mappings.OUTSIDE: "o",
                         mappings.UNKNOWN: "u"}

step1_target_decode_dict = {mappings.MEMBRANE: "M",
                            mappings.NONMEMBRANE: "n"}


class TMSEGEncoder:
    def encode_sequence(self, sequence):
        return [sequence_encode_dict[x] for x in sequence]

    def encode_structure(self, structure):
        return [structure_encode_dict[z] for z in structure]


class TMSEGDecoder:

    # Wrong!! Decoding of seqence doesn't work right now
    def decode_sequence(self, sequence):
        return "".join([observables[x] for x in sequence])

    def decode_structure(self, structure):
        # return "".join([structure_decode_dict[z] for z in structure])
        return "".join([step1_target_decode_dict[z] for z in structure])
