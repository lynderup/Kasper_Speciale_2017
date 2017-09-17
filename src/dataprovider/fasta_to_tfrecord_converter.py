import tensorflow as tf

import dataprovider.read_fasta as read_fasta


observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

INSIDE = 1
HELIX = 2
OUTSIDE = 3

structure_dict = {"1": INSIDE,
                  "H": HELIX,
                  "h": HELIX,
                  "2": OUTSIDE,
                  "U": 0,
                  "0": 0,
                  "L": 0}


def fasta_entry_to_example(fasta_entry):
    entry_parts = fasta_entry.split("\n")

    # sequence = np.asarray([observables.index(x) for x in entry_parts[1]])
    sequence = [observables.index(x) for x in entry_parts[1]]
    # structure = np.asarray([structure_dict[z] for z in entry_parts[2]])
    structure = [structure_dict[z] for z in entry_parts[2]]

    feature = {'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=sequence)),
               'structure': tf.train.Feature(int64_list=tf.train.Int64List(value=structure))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def fasta_to_tfrecord(path, fasta_path, tfrecord_path, sets):

    def fasta_filename(set): return "".join([path, fasta_path, set, ".fasta"])

    def tfrecord_filename(set): return "".join([path, tfrecord_path, set, ".tfrecord"])

    for set in sets:
        fasta_entries = read_fasta.read_fasta(fasta_filename(set))

        with tf.python_io.TFRecordWriter(tfrecord_filename(set)) as writer:

            for entry in fasta_entries:
                example = fasta_entry_to_example(entry)

                writer.write(example.SerializeToString())


if __name__ == '__main__':
    fasta_to_tfrecord()