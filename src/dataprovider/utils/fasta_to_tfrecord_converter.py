import tensorflow as tf

import dataprovider.utils.read_fasta as read_fasta
import dataprovider.utils.pssm_reader as read_pssm

from encoders.tmseg_encoder import TMSEGEncoder


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def list_feature(features):
    feature_list = []

    for feature in features:
        if type(feature) == list:
            feature_list.append(int_feature(feature))
        else:
            feature_list.append(int_feature([feature]))

    return tf.train.FeatureList(feature=feature_list)


def fasta_entry_to_example(encoder, fasta_entry):
    entry_parts = fasta_entry.split("\n")

    name = entry_parts[0].split("|")[0]
    sequence = encoder.encode_sequence(entry_parts[1])
    structure = encoder.encode_structure(entry_parts[2])

    pssm = read_pssm.read_pssm_file(name)

    # sequence = [observables.index(x) for x in entry_parts[1]]
    # structure = [structure_dict[z] for z in entry_parts[2]]

    context = {'length': int_feature([len(sequence)]),
               'name': bytes_feature([name.encode()])}

    feature_list = {'sequence': list_feature(sequence),
                    'structure': list_feature(structure),
                    'pssm': list_feature(pssm)}

    example = tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                       feature_lists=tf.train.FeatureLists(feature_list=feature_list))

    return example


def fasta_to_tfrecord(path, fasta_path, tfrecord_path, sets, encoder):

    def fasta_filename(set): return "".join([path, fasta_path, set, ".fasta"])

    def tfrecord_filename(set): return "".join([path, tfrecord_path, set, ".tfrecord"])

    for set in sets:
        fasta_entries = read_fasta.read_fasta(fasta_filename(set))

        with tf.python_io.TFRecordWriter(tfrecord_filename(set)) as writer:

            for entry in fasta_entries:
                example = fasta_entry_to_example(encoder, entry)
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    path = "datasets/tmseg/data/sets/unmasked_hval0/"
    fasta_entries = read_fasta.read_fasta(path + "opm_set1.fasta")

    encoder = TMSEGEncoder()
    [fasta_entry_to_example(encoder, fasta_entry) for fasta_entry in fasta_entries]