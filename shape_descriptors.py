import tensorflow as tf
from keras.layers import Layer


@tf.function
def soft_size(probabilities: tf.Tensor):
    rank = len(tf.shape(probabilities))
    return tf.reduce_sum(probabilities, tuple(range(1, rank - 1)))


@tf.function
def soft_dist_centroid(probabilities: tf.Tensor):
    """
    computes the distance to the centroid based on the probability map.
    Adapted from pytorch code from:
        https://github.com/HKervadec/shape_descriptors/
    """
    shape = tf.shape(probabilities)
    b = shape[0]
    h = shape[1]
    w = shape[2]
    k = shape[3]
    dim = 2
    if len(shape) > 4:
        raise ValueError("Input should be a 4 dimensional tensor")

    interval0 = tf.linspace(0.0, tf.cast(h - 1, tf.float32), h)
    interval1 = tf.linspace(0.0, tf.cast(w - 1, tf.float32), w)
    grids = tf.meshgrid(interval0, interval1)
    grids = tf.stack(grids, 2)

    sizes = tf.einsum("bhwk->bk", probabilities)
    sizes = tf.expand_dims(sizes, -1)

    centroids = tf.einsum("bhwk,hwl->bkl", probabilities,
                          grids) / (sizes + 1e-10)

    diffs = tf.reshape(grids, (1, h, w, 1, dim)) - tf.reshape(
        centroids, (b, 1, 1, k, dim)
    )

    dist_centroid = tf.einsum("bhwk,bhwkl->bkl", probabilities, diffs**2)
    dist_centroid = tf.sqrt(dist_centroid)

    return dist_centroid


if __name__ == "__main__":
    import numpy as np

    probs = np.zeros([1, 50, 50, 3], np.float32)
    probs[0, 10:20, 10:20, 0] = 1.0
    probs[0, 25:30, 25:30, 1] = 1.0
    probs[..., 2] = 1.0 - probs[..., 0] - probs[..., 1]

    # test soft_dist_centroid
    sdc = soft_dist_centroid(probs)
    sdc = np.array(sdc)
    assert sdc.shape == (1, 3, 2)
    assert sdc[0, 2, 0] > sdc[0, 0, 0] > sdc[0, 1, 0]

    print(soft_size(probs))
