import model.joint_model as joint_model
import model.util as util
from decoder.decode_and_print_step1 import decode_and_print_step1
from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGDecoder

if __name__ == '__main__':

    m = joint_model.Model(logdir="test/", should_step3=False)

    # m.train()
    # predictions = m.inference()
    # set_lengths, set_inputs, set_targets, set_predictions = m.inference()
    #
    # corrected_predictions = util.numpy_step2(set_predictions)
    # predictions = zip(set_lengths, set_inputs, set_targets, set_predictions, corrected_predictions)
    #
    # decoder = TMSEGDecoder()
    # decode_and_print_step1(predictions, decoder)


