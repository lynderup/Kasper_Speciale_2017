import dataprovider.mappings as mappings

sequence_decode_dict = {mappings.ALA: "A",
                        mappings.ARG: "R",
                        mappings.ASN: "N",
                        mappings.ASP: "D",
                        mappings.CYS: "C",
                        mappings.GLU: "E",
                        mappings.GLN: "Q",
                        mappings.GLY: "G",
                        mappings.HIS: "H",
                        mappings.ILE: "I",
                        mappings.LEU: "L",
                        mappings.LYS: "K",
                        mappings.MET: "M",
                        mappings.PHE: "F",
                        mappings.PRO: "P",
                        mappings.SER: "S",
                        mappings.THR: "T",
                        mappings.TRP: "W",
                        mappings.TYR: "Y",
                        mappings.VAL: "V"}

structure_decode_dict = {mappings.INSIDE: "i",
                         mappings.HELIX: "h",
                         mappings.OUTSIDE: "o",
                         mappings.UNKNOWN: "u"}

step1_target_decode_dict = {mappings.MEMBRANE: "M",
                            mappings.NONMEMBRANE: "n"}


def decode_sequence(sequence):
    return "".join([sequence_decode_dict[x] for x in sequence])


def decode_step1_targets(structure):
    # return "".join([structure_decode_dict[z] for z in structure])
    return "".join([step1_target_decode_dict[z] for z in structure])


def cut_to_lengths(length, inputs, targets, predictions):
    return inputs[0:length], targets[0:length], predictions[0:length]


def decode(inputs, targets, predictions):
    return decode_sequence(inputs), \
           decode_step1_targets(targets), \
           decode_step1_targets(predictions),


def print_predictions(inputs, targets, predictions):
    print(inputs)
    print(targets)
    print(predictions)


def decode_step123(predictions):

    decoded_predictions = []

    for prediction in predictions:
        prediction = cut_to_lengths(*prediction)
        prediction = decode(*prediction)

        decoded_predictions.append(("", *prediction))

    return decoded_predictions
