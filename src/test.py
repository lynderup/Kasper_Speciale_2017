import tensorflow as tf
from dataprovider.tmseg_dataset_provider import TMSEGDatasetProvider

dataprovider = TMSEGDatasetProvider(batch_size=10)
handle, iterator = dataprovider.get_iterator()

lengths, sequences, structures_step1, structures_step3 = iterator.get_next()

with tf.Session() as sess:

    sess.run(dataprovider.get_table_init_op())

    test_handle, _ = sess.run(dataprovider.get_test_iterator_handle())  # get_handle returns (handle, init_op)
    test_feed = {handle: test_handle}

    step1, step3 = sess.run([structures_step1, structures_step3], feed_dict=test_feed)

    print(step1[0].tolist())
    print(step3[0].tolist())