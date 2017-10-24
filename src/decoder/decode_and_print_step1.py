from decoder.write_to_fasta import write_to_fasta


def cut_to_lengths(length, inputs, targets, predictions, corrected_predictions):
    return inputs[0:length], targets[0:length], predictions[0:length], corrected_predictions[0:length]


def decode(inputs, targets, predictions, corrected_predictions, decoder):
    return decoder.decode_sequence(inputs),\
           decoder.decode_structure(targets),\
           decoder.decode_structure(predictions),\
           decoder.decode_structure(corrected_predictions)


def print_predictions(inputs, targets, predictions, corrected_predictions):
    print(inputs)
    print(targets)
    print(predictions)
    print(corrected_predictions)


def decode_and_print_step1(predictions, decoder):

    for prediction in predictions:
        prediction = cut_to_lengths(*prediction)
        prediction = decode(*prediction, decoder)
        print_predictions(*prediction)
