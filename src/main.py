import model

from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGDecoder

from decoder.decode_and_print_step1 import decode_and_print_step1

if __name__ == '__main__':
    model.train()
    predictions = model.inference()

    decoder = TMSEGDecoder()
    decode_and_print_step1(predictions, decoder)


