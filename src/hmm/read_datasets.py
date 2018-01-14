
def read_sequence(file_part):
    parts = file_part.split("\n")
    return parts[0], parts[1], parts[2]


def read_datasets(filenames):

    datasets = []

    for filename in filenames:
        with open(filename, "r") as file:

            file_parts = file.read().split(">")

            test_sequences = []

            for file_part in file_parts:
                if len(file_part) == 0:
                    continue

                test_sequences.append(read_sequence(file_part))

            datasets.append(test_sequences)

    return datasets