import time

from hmm.read_datasets import read_datasets
import hmm.models as models

from evaluaters.statistics import Statistics

dataset_path = "datasets/tmseg/data/sets/unmasked_hval0/"
sets = ["opm_set1", "opm_set2", "opm_set3", "opm_set4"]


filenames = [dataset_path + setname + ".fasta" for setname in sets]


def train_and_test_model(model_construct_function, model, use_psedo_count=False):

    runs = []
    for i in range(0, 10):

        datasets = read_datasets(filenames[0:3])
        start = time.time()
        hmm = model_construct_function(datasets, use_psedo_count)
        training_time = time.time() - start

        predictions = []

        start = time.time()
        for name, xs, zs in read_datasets([filenames[3]])[0]:
            log_likelihood, decoding = hmm.viterbi_decoding(xs)

            predictions.append((name, xs, zs, decoding))
        decoding_time = time.time() - start

        print("Training time: %s" % training_time)
        print("Decoding time: %s" % decoding_time)

        runs.append(predictions)

    return runs


def do_hmm():
    return train_and_test_model(models.construct_4_state_model, "4 state model with psedo count", True)

if __name__ == '__main__':
    statistics = Statistics()

    runs = train_and_test_model(models.construct_4_state_model, "4 state model with psedo count", True)

    statistics.add_model(("HMM", runs))

    statistics.print_predictions()
    statistics.print_statistics()