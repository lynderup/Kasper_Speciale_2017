import model

from dataprovider.tmseg_dataset_provider import TMSEGDatasetProvider
from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGDecoder
from decoder.decode_and_print_step1 import decode_and_print_step1

if __name__ == '__main__':
    dataprovider = TMSEGDatasetProvider()
    config = model.ModelConfig()
    m = model.Model(dataprovider, config)

    m.train()
    predictions = m.inference()

    decoder = TMSEGDecoder()
    decode_and_print_step1(predictions, decoder)


