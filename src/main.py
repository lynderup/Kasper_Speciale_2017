import model.joint_model as joint_model
import model.util as util
import decoder.decoder as decoder
import model.hyper_params_search as hyper_params_search
import model.cross_validation as cross_validation

from evaluaters.statistics import Statistics

config = joint_model.StepConfig(batch_size=10,
                                num_input_classes=20,
                                num_output_classes=2,
                                starting_learning_rate=0.01,
                                decay_steps=10,
                                decay_rate=0.99,
                                num_units=50,  # 50
                                train_steps=1000)  # 1000

model_config = joint_model.default_config._replace(step1_config=config)

def test():
    statistics = Statistics()

    # m = joint_model.Model(logdir="test/", should_step3=False)
    #
    # m.train()
    # set_lengths, set_inputs, set_targets, set_predictions = m.inference()

    runs = cross_validation.do_3_fold_cross_validation(config=model_config)

    step1_predictions = []
    step2_predictions = []

    for set_lengths, set_inputs, set_targets, set_predictions in runs:
        predictions = zip(set_lengths, set_inputs, set_targets, set_predictions)
        step1_predictions.append(decoder.decode_step123(predictions))

        corrected_predictions = util.numpy_step2(set_predictions)
        predictions = zip(set_lengths, set_inputs, set_targets, corrected_predictions)
        step2_predictions.append(decoder.decode_step123(predictions))

    statistics.add_model(("Step1", step1_predictions))
    statistics.add_model(("Step2", step2_predictions))

    statistics.print_predictions()
    statistics.print_statistics()

if __name__ == '__main__':
    test()
    # hyper_params_search.do_hyper_params_search()
    # cross_validation.do_3_fold_cross_validation()



