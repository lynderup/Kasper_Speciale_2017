import numpy as np
import math

import dataprovider.dataset_provider as dataset_provider

def find_membranes(logits):
    batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
    batch_membranes = []

    for prediction in batch_predictions:

        in_membrane = False
        membranes = []
        membrane_start_index = -1

        for i, z in enumerate(prediction):
            if z == dataset_provider.MEMBRANE:
                if not in_membrane:
                    in_membrane = True
                    membrane_start_index = i
            else:
                if in_membrane:
                    in_membrane = False
                    membranes.append((membrane_start_index, i))

        batch_membranes.append(np.asarray(membranes))

    return np.asarray(batch_membranes)


def filter_membranes(batch_membranes):

    new_batch_membranes = []
    for membranes in batch_membranes:

        new_membranes = []
        for start, end in membranes:
            length = end - start

            if length > 5:
                if length >= 35:
                    new_membranes.append((start, math.floor(length / 2)))
                    new_membranes.append((math.ceil(length / 2), end))
                else:
                    new_membranes.append((start, end))
        new_batch_membranes.append(np.asarray(new_membranes))
    return np.asarray(new_batch_membranes)


def numpy_step2(logits):
    batch_predictions = np.swapaxes(np.argmax(logits, axis=2), 0, 1)
    batch_membranes = find_membranes(logits)

    new_predictions = []

    for i, membranes in enumerate(batch_membranes):
        prediction = batch_predictions[i]
        for start, end in membranes:
            length = end - start

            if length <= 5:
                prediction[start:end] = [dataset_provider.NOTMEMBRANE] * length

            if length >= 35:
                new_membrane = [dataset_provider.MEMBRANE] * length
                new_membrane[math.floor(length / 2)] = dataset_provider.NOTMEMBRANE
                prediction[start:end] = new_membrane

        new_predictions.append(prediction)

    return np.asarray(new_predictions)
