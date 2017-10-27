import tensorflow as tf


def find_endpints(batch):

    end_points_batch = []
    for sequence in batch:

        in_ones = False
        end_points = []

        for i, x in enumerate(sequence):
            if x == 1:
                if not in_ones:
                    in_ones = True
                    end_points.append(i)
            else:
                if in_ones:
                    in_ones = False
                    end_points.append(i - 1)

        end_points_batch.append(end_points)

    return end_points_batch

def test_map(endpoint):
    return tf.squeeze(tf.slice(a, [a_dim, tf.minimum(endpoint - 1, tf.shape(a)[1] - 3)], [1, 3]))

a = tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])

# b = tf.constant([[[0, 4], [0, 5], [0, 6]], [[1, 2], [1, 3], [1, 4]]])
# c = tf.gather_nd(params=a, indices=b)

endpoints_t = tf.placeholder(tf.int32, shape=[None])
a_dim = tf.placeholder(tf.int32, shape=[])

slices = tf.map_fn(test_map, endpoints_t)

with tf.Session() as sess:

    input = sess.run(a)
    endpoints = find_endpints(input)

    print(endpoints)
    for i, e in enumerate(endpoints):
        feed_dict = {endpoints_t: e,
                     a_dim: i}
        out = sess.run(slices, feed_dict=feed_dict)
        print(out)

    # print(sess.run(endpoints_t, feed_dict={endpoints_t:endpoints[0]}))
