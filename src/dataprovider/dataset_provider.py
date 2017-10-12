import tensorflow as tf

# Structure constants
INSIDE = 1
HELIX = 2
OUTSIDE = 3
UNKNOWN = 0

# Step 1 target constants
MEMBRANE = 0
NOTMEMBRANE = 1


structure_to_step1_target_dict = {INSIDE: NOTMEMBRANE,
                                  HELIX: MEMBRANE,
                                  OUTSIDE: NOTMEMBRANE,
                                  UNKNOWN: NOTMEMBRANE}

keys = tf.convert_to_tensor((INSIDE, HELIX, OUTSIDE, UNKNOWN), dtype=tf.int64)
values = tf.convert_to_tensor((NOTMEMBRANE, MEMBRANE, NOTMEMBRANE, NOTMEMBRANE), dtype=tf.int64)
table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)


def structure_to_step_targets(lengths, sequence, structure):
    # step1_target = tf.map_fn(lambda s: structure_to_step1_target_dict[s], structure)
    step1_target = table.lookup(structure)

    return lengths, sequence, step1_target


class DatasetProvider:

    def get_table_init_op(self):
        return table.init
