# from decoder.write_to_fasta import write_to_fasta

import evaluaters.compare_prediction as compare_prediction


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

    decoded_predictions = []
    decoded_corrected_predictions = []

    for prediction in predictions:
        prediction = cut_to_lengths(*prediction)
        prediction = decode(*prediction, decoder)
        print_predictions(*prediction)

        inputs, targets, _predictions, corrected_predictions = prediction

        decoded_predictions.append(("", inputs, targets, _predictions))
        decoded_corrected_predictions.append(("", inputs, targets, corrected_predictions))

    print("Endpoint below 5")
    precision, recall = compare_prediction.compare_predictions(decoded_predictions,
                                                                 compare_prediction.endpoints_diff_below_5_overlap_over_50_percent)
    print("Precision: %.4f Recall: %.4f" % (precision, recall))

    precision, recall = compare_prediction.compare_predictions(decoded_corrected_predictions,
                                                                 compare_prediction.endpoints_diff_below_5_overlap_over_50_percent)
    print("Precision: %.4f Recall: %.4f" % (precision, recall))

    print("overlap 25%")
    precision, recall = compare_prediction.compare_predictions(decoded_predictions,
                                                               compare_prediction.overlap_over_25_percent)
    print("Precision: %.4f Recall: %.4f" % (precision, recall))

    precision, recall = compare_prediction.compare_predictions(decoded_corrected_predictions,
                                                               compare_prediction.overlap_over_25_percent)
    print("Precision: %.4f Recall: %.4f" % (precision, recall))
