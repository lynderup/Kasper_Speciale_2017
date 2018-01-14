import numpy as np
from hmm.hmm import HMM


def states_to_indexs(states, list_of_states):
    return [states.index(state) for state in list_of_states]


def train_by_counting(number_of_hidden, number_of_observables, training_data, use_psedo_count):

    if use_psedo_count:
        initials_count = np.ones(number_of_hidden)
        transitions_count = np.ones((number_of_hidden, number_of_hidden))
        emissions_count = np.ones((number_of_hidden, number_of_observables))
        # transitions_count[2, 0] = 0
        # transitions_count[0, 2] = 0
    else:
        initials_count = np.zeros(number_of_hidden)
        transitions_count = np.zeros((number_of_hidden, number_of_hidden))
        emissions_count = np.zeros((number_of_hidden, number_of_observables))

    for dataset in training_data:
        for name, xs, zs in dataset:

            initials_count[zs[0]] += 1
            emissions_count[zs[0], xs[0]] += 1

            last_z = zs[0]

            for x, z in zip(xs[1:], zs[1:]):
                emissions_count[z, x] += 1
                transitions_count[last_z, z] += 1
                last_z = z

    # print(initials_count)
    # print(transitions_count)
    # print(emissions_count)

    initials = initials_count / np.sum(initials_count)
    transitions = np.zeros_like(transitions_count)
    emissions = np.zeros_like(emissions_count)

    for idx, row in enumerate(transitions_count):
        transitions[idx] = row / np.sum(row)

    for idx, row in enumerate(emissions_count):
        emissions[idx] = row / np.sum(row)

    # print(initials)
    # print(transitions)
    # print(emissions)

    return initials, transitions, emissions


def construct_3_state_model(training_data, use_psedo_count=False):

    hidden = ['i', 'M', 'o']
    observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

    transformed_training_data = []

    for dataset in training_data:
        transformed_dataset = []

        for name, xs, zs in dataset:

            xs = states_to_indexs(observables, xs)
            zs = states_to_indexs(hidden, zs)

            transformed_dataset.append((name, xs, zs))

        transformed_training_data.append(transformed_dataset)

    initials, transitions, emissions = train_by_counting(len(hidden), len(observables), transformed_training_data, use_psedo_count)

    return HMM(hidden, observables, initials, transitions, emissions, hidden)


def construct_4_state_model(training_data, use_psedo_count=False):

    observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

    transformed_training_data = []

    for dataset in training_data:
        transformed_dataset = []

        for name, xs, zs in dataset:

            xs = states_to_indexs(observables, xs)

            transformed_zs = []

            out_going = True

            if zs[0] == "2":
                out_going = False

            for z in zs:
                if z == "1":
                    transformed_zs.append(0)
                    out_going = True
                elif z == "H":
                    if out_going:
                        transformed_zs.append(1)
                    else:
                        transformed_zs.append(3)
                elif z == "2":
                    transformed_zs.append(2)
                    out_going = False
                elif z == "U" or z == "0":
                    transformed_zs.append(0)
                else:
                    transformed_zs.append(transformed_zs[-1])

            transformed_dataset.append((name, xs, transformed_zs))

        transformed_training_data.append(transformed_dataset)

    initials, transitions, emissions = train_by_counting(4, len(observables), transformed_training_data, use_psedo_count)

    return HMM([0, 1, 2, 3], observables, initials, transitions, emissions, ['1', 'H', '2', 'H'])






