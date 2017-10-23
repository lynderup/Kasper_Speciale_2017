import model

from dataprovider.tmseg_dataset_provider import TMSEGDatasetProvider
from encoders_and_decoders.tmseg_encoder_and_decoder import TMSEGDecoder
from decoder.decode_and_print_step1 import decode_and_print_step1

if __name__ == '__main__':
    config = model.ModelConfig()
    dataprovider = TMSEGDatasetProvider(batch_size=config.batch_size)
    m = model.Model(dataprovider, config, "test/")

    m.train()
    predictions = m.inference()

    decoder = TMSEGDecoder()
    decode_and_print_step1(predictions, decoder)


