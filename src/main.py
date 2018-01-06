import model.joint_model as joint_model
import model.util as util
import decoder.decoder as decoder

from evaluaters.statistics import Statistics

if __name__ == '__main__':

    statistics = Statistics()

    m = joint_model.Model(logdir="test/", should_step3=False)

    m.train()
    set_lengths, set_inputs, set_targets, set_predictions = m.inference()

    step1_predictions = zip(set_lengths, set_inputs, set_targets, set_predictions)
    statistics.add_model(("Step1", decoder.decode_step123(step1_predictions)))

    corrected_predictions = util.numpy_step2(set_predictions)
    step2_predictions = zip(set_lengths, set_inputs, set_targets, corrected_predictions)
    statistics.add_model(("Step2", decoder.decode_step123(step2_predictions)))

    statistics.print_statistics()




