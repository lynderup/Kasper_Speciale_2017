import dataprovider.mappings as mappings
import decoder.decoder as decoder

import evaluaters.compare_tm_pred as compare_char


def endpoints_diff_below_5_overlap_over_50_percent(start_diff, end_diff, overlap, longest):
    return start_diff <= 5 and end_diff <= 5 and overlap * 2 >= longest


def overlap_over_25_percent(start_diff, end_diff, overlap, longest):
    return overlap * 4 >= longest


def overlap_over_x(x):
    ratio = 100.0 / x

    return lambda start_diff, end_diff, overlap, longest: overlap * ratio >= longest


def endpoint_diff_below_x(x):

    return lambda start_diff, end_diff, overlap, longest: start_diff <= x and end_diff <= x


def find_tmh(annotation):

    tmh = []
    is_in_tmh = False
    tmh_start = 0

    for i, z in enumerate(annotation):

        if z == decoder.step1_target_decode_dict[mappings.MEMBRANE]:
            if not is_in_tmh:
                is_in_tmh = True
                tmh_start = i
        else:
            if is_in_tmh:
                is_in_tmh = False
                tmh.append((tmh_start, i - 1))

    return tmh


def compare_prediction(name, true, pred, measurement):

    true_tmh = find_tmh(true)
    pred_tmh = find_tmh(pred)

    number_of_predicted_tmh = len(pred_tmh)
    number_of_observed_tmh = len(true_tmh)
    number_of_correct_predictions = 0

    for start_pred, end_pred in pred_tmh:
        for start_true, end_true in true_tmh:

            start_diff = abs(start_pred - start_true)
            end_diff = abs(end_pred - end_true)

            true_length = end_true - start_true
            pred_length = end_pred - start_pred

            overlap = min(end_pred, end_true) - max(start_pred, start_true)
            longest = max(true_length, pred_length)

            if measurement(start_diff, end_diff, overlap, longest):
                number_of_correct_predictions += 1
                break

    return number_of_correct_predictions, number_of_predicted_tmh, number_of_observed_tmh


def compare_predictions(predictions, measurement, should_print=False):

    if measurement == "char_compare":
        return char_compare(predictions)

    number_of_correct_predictions = 0
    number_of_predicted_tmh = 0
    number_of_observed_tmh = 0

    for name, xs, zs, ps in predictions:
        correct_predictions, predicted_tmh, observed_tmh = compare_prediction(name, zs, ps, measurement)

        number_of_correct_predictions += correct_predictions
        number_of_predicted_tmh += predicted_tmh
        number_of_observed_tmh += observed_tmh

    if number_of_predicted_tmh != 0:
        precision = number_of_correct_predictions / number_of_predicted_tmh
    else:
        precision = 0

    if number_of_observed_tmh != 0:
        recall = number_of_correct_predictions / number_of_observed_tmh
    else:
        # There should always be observed tmh
        recall = 1

    return precision, recall


def to_dictionary(predictions):
    true = {}
    pred = {}

    for i, (name, xs, zs, prediction) in enumerate(predictions):
        true[i] = "%s # %s" % (xs, zs)
        pred[i] = "%s # %s" % (xs, prediction)

    return true, pred


def char_compare(predictions):
    true, pred = to_dictionary(predictions)

    ac, sn, sp = compare_char.do_compare(true, pred, False)
    precision = sp
    recall = sn

    return precision, recall
