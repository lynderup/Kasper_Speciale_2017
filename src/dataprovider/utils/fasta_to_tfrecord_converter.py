import tensorflow as tf

import dataprovider.utils.read_fasta as read_fasta


def fasta_entry_to_example(encoder, fasta_entry):
    entry_parts = fasta_entry.split("\n")

    sequence = encoder.encode_sequence(entry_parts[1])
    structure = encoder.encode_structure(entry_parts[2])

    # sequence = [observables.index(x) for x in entry_parts[1]]
    # structure = [structure_dict[z] for z in entry_parts[2]]

    context = {'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sequence)]))}

    sequence_feature_list = []
    structure_feature_list = []

    for seq, struc in zip(sequence, structure):
        sequence_feature_list.append(tf.train.Feature(int64_list=tf.train.Int64List(value=[seq])))
        structure_feature_list.append(tf.train.Feature(int64_list=tf.train.Int64List(value=[struc])))

    feature_list = {'sequence': tf.train.FeatureList(feature=sequence_feature_list),
                    'structure': tf.train.FeatureList(feature=structure_feature_list)}

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
    fasta_to_tfrecord()