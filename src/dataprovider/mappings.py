import tensorflow as tf

# Sequence constants
ALA = 0  # A Alanine
ARG = 2  # R Arginine
ASN = 1  # N Asparagine
ASP = 3  # D Aspartic Acid
CYS = 4  # C Cysteine
GLU = 5  # E Glutamic Acid
GLN = 6  # Q Glutamine
GLY = 7  # G Glycine
HIS = 8  # H Histidine
ILE = 9  # I Isoleucine
LEU = 10  # L Leucine
LYS = 11  # K Lysine
MET = 12  # M Methionine
PHE = 13  # F Phenylalanine
PRO = 14  # P Proline
SER = 15  # S Serine
THR = 16  # T Threonine
TRP = 17  # W Tryptophan
TYR = 18  # Y Tyrosine
VAL = 19  # V Valine

# Structure constants
INSIDE = 1
HELIX = 2
OUTSIDE = 3
LOOP = 4
UNKNOWN = 0

# Step 1 target constants
MEMBRANE = 0
NONMEMBRANE = 1

structure_to_step1_target_dict = {INSIDE: NONMEMBRANE,
                                  HELIX: MEMBRANE,
                                  OUTSIDE: NONMEMBRANE,
                                  LOOP: NONMEMBRANE,
                                  UNKNOWN: NONMEMBRANE}

# hydrophobicity, charge, polarity
sequence_to_sup_data_dict = {ALA: (1.8, 0, 0),
                             ARG: (-4.5, 1, 1),
                             ASN: (-3.5, 0, 1),
                             ASP: (-3.5, -1, 1),
                             CYS: (2.5, 0, 1),
                             GLU: (-3.5, -1, 1),
                             GLN: (-3.5, 0, 1),
                             GLY: (-0.4, 0, 0),
                             HIS: (-3.2, 0, 1),
                             ILE: (4.5, 0, 0),
                             LEU: (3.8, 0, 0),
                             LYS: (-3.9, 1, 1),
                             MET: (1.9, 0, 0),
                             PHE: (2.8, 0, 0),
                             PRO: (-1.6, 0, 0),
                             SER: (-0.8, 0, 1),
                             THR: (-0.7, 0, 1),
                             TRP: (-0.9, 0, 0),
                             TYR: (-1.3, 0, 1),
                             VAL: (4.2, 0, 0)}


def dict_to_hashtable(dict):
    keys = []
    values = []

    for key, value in dict.items():
        keys.append(key)
        values.append(value)

    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(keys, dtype=tf.int64),
                                                    tf.convert_to_tensor(values, dtype=tf.int64)), -1)


def dict_to_embedding_tensor(dict):

    embedding = []
    for i in range(len(sequence_to_sup_data_dict)):
        embedding.append(sequence_to_sup_data_dict[i])

    embedding_tensor = tf.convert_to_tensor(embedding, dtype=tf.float32)
    return embedding_tensor

    # keys = tf.convert_to_tensor((INSIDE, HELIX, OUTSIDE, UNKNOWN), dtype=tf.int64)
    # values = tf.convert_to_tensor((NOTMEMBRANE, MEMBRANE, NOTMEMBRANE, NOTMEMBRANE), dtype=tf.int64)
    # table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
