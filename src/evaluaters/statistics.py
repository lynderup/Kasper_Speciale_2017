import evaluaters.compare_prediction as compare_prediction


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

            for model_name, model in self.models:

                print(model_name)
                precision, recall = compare_prediction.compare_predictions(model,
                                                                           evaluator)
                print("Precision: %.4f Recall: %.4f" % (precision, recall))

    # Assumes predictions in same order in all added models
    def print_predictions(self):

        models = [model for _, model in self.models]
        if len(models) > 0:
            for i, (name, sequence, targets, _) in enumerate(models[0]):
                print(name)
                print(sequence)
                print(targets)

                for model in models:
                    print(model[i][3])




