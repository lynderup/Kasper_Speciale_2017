import tensorflow as tf

import dataprovider.dataprovider_step3 as dataprovider_step3

dataprovider = dataprovider_step3.DataproviderStep3(10)

handle, iterator = dataprovider.get_iterator()
lengths, sequences, sequence_sup_data, targets_step3 = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(dataprovider.get_table_init_op())

    train_handle, _ = sess.run(dataprovider.get_train_iterator_handle())
    train_feed = {handle: train_handle}
    fetches = [lengths, sequences, sequence_sup_data, target_step3s]

    lengths, sequences, sequence_sup_data, target_step3s = sess.run(fetches=fetches, feed_dict=train_feed)

    print(lengths)
    # print(sequences)
    # print(sequence_sup_data)
    print(target_step3)


