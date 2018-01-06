import evaluaters.compare_prediction as compare_prediction
import numpy as np

class Statistics:

    def __init__(self):
        self.evaluators = []
        self.models = []

        self.evaluators.append(("Endpoint below 5", compare_prediction.endpoints_diff_below_5_overlap_over_50_percent))
        self.evaluators.append(("Overlap 25%", compare_prediction.overlap_over_25_percent))

    def add_model(self, model):
        self.models.append(model)

    def print_statistics(self):

        for evaluator_name, evaluator in self.evaluators:
            print(evaluator_name)

            for model_name, runs in self.models:

                print(model_name)

                precisions = []
                recalls = []

                for run in runs:
                    precision, recall = compare_prediction.compare_predictions(run, evaluator)
                    precisions.append(precision * 100)
                    recalls.append(recall * 100)

                precision_mean = np.mean(precisions)
                precision_var = np.var(precisions)

                recall_mean = np.mean(recalls)
                recall_var = np.var(recalls)

                print("Precision: %.4f %.4f Recall: %.4f %.4f" % (precision_mean,
                                                                  precision_var,
                                                                  recall_mean,
                                                                  recall_var))
                print(precisions)
                print(recalls)

    # Assumes predictions in same order in all added models
    def print_predictions(self):

        models = [model[0] for _, model in self.models]
        if len(models) > 0:
            for i, (name, sequence, targets, _) in enumerate(models[0]):
                print(name)
                print(sequence)
                print(targets)

                for model in models:
                    print(model[i][3])




