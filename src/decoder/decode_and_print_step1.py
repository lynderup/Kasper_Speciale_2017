def cut_to_lengths(length, inputs, targets, predictions):
    return inputs[0:length], targets[0:length], predictions[0:length]


def decode(inputs, targets, predictions, decoder):
    return decoder.decode_sequence(inputs),\
           decoder.decode_structure(targets),\
           decoder.decode_structure(predictions)


def print_predictions(inputs, targets, predictions):
    print(inputs)
    print(targets)
    print(predictions)


def decode_and_print_step1(predictions, decoder):

    for prediction in predictions:
        prediction = cut_to_lengths(*prediction)
        prediction = decode(*prediction, decoder)
        print_predictions(*prediction)
