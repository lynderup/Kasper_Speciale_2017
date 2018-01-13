import model.joint_model as joint_model
import decoder.decoder as decoder
import evaluaters.compare_prediction as compare_prediction
import model.cross_validation as cross_validation

import numpy as np

# params_to_search_coarse = {"starting_learning_rate": [1.0, 0.1, 0.01, 0.001],
#                            "decay_rate": [0.99, 0.96, 0.92],
#                            "num_units": [5, 10, 20, 50],
#                            "train_steps": [100, 200, 500, 1000]}

params_to_search_coarse = {"l2_beta": [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]}


def do_hyper_params_search():
    step1_config = joint_model.default_step1_config
    config = joint_model.default_config

    # step1_config = step1_config._replace(train_steps=10)

    has_changed = True

    while has_changed:
        has_changed = False

        for key, values in params_to_search_coarse.items():

            print("Searching %s" % key)

            value_precision_pairs = []
            value_before = getattr(step1_config, key)

            for value in values:
                print("Trying %s = %s" % (key, value))
                step1_config = step1_config._replace(**{key: value})
                config = config._replace(step1_config=step1_config)

                # m = joint_model.Model(logdir="hyper_params_search/", config=config, should_step3=False)
                #
                # m.train()
                # set_lengths, set_inputs, set_targets, set_predictions = m.inference()

                runs = cross_validation.do_3_fold_cross_validation(logdir="hyper_params_search/", config=config)

                step1_predictions = []
                for set_lengths, set_inputs, set_targets, set_predictions in runs:
                    run = zip(set_lengths, set_inputs, set_targets, set_predictions)
                    decoded_step1_predictions = decoder.decode_step123(run)
                    step1_predictions.append(decoded_step1_predictions)

                precisions = []
                recalls = []

                for predictions in step1_predictions:
                    precision, recall = compare_prediction.compare_predictions(predictions,
                                                                               compare_prediction.overlap_over_25_percent)
                    precisions.append(precision)
                    recalls.append(recall)

                mean_precision = np.mean(precisions)
                mean_recall = np.mean(recalls)
                value_precision_pairs.append((value, np.mean([mean_precision, mean_recall])))

            best_value = value_precision_pairs[np.argmax(value_precision_pairs, axis=0)[1]][0]
            step1_config = step1_config._replace(**{key: best_value})

            if best_value != value_before:
                has_changed = True

            print("Best value for %s is %s" % (key, best_value))
        print("Best found config:")
        print(step1_config)


