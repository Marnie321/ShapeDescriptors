import tensorflow as tf
from keras.layers import Layer


@tf.function
def soft_size(probabilities: tf.Tensor):
    rank = len(tf.shape(probabilities))
    return tf.reduce_sum(probabilities, tuple(range(1, rank-1)))


@tf.function
def soft_dist_centroid(probabilities):
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


@tf.function
def gradient2d(image, st_element):
    """
    Gradient Operator
    """
    dilation = tf.nn.dilation2d(image,
                                st_element,
                                (1, 1, 1, 1),
                                'SAME',
                                'NHWC',
                                (1, 1, 1, 1))
    erosion = tf.nn.erosion2d(image,
                              st_element,
                              (1, 1, 1, 1),
                              'SAME',
                              'NHWC',
                              (1, 1, 1, 1))
    return dilation - erosion


class CurvatureEstimator(Layer):
    def __init__(self, num_classes, neighborhood_size, **kwargs):
        super().__init__(**kwargs)

        self.neighborhood_size = neighborhood_size
        self.filter_size = 2 * neighborhood_size + 1
        self.num_classes = num_classes

        kernel = tf.linspace(-neighborhood_size,
                             neighborhood_size,
                             self.filter_size)
        kernel = tf.cast(kernel, tf.float32)
        kernel = tf.expand_dims(kernel, 1)
        kernel = tf.tile(kernel, (1, num_classes))
        self.kernel_x = tf.reshape(kernel,
                                   (self.filter_size, 1, num_classes, 1))
        self.kernel_y = tf.reshape(kernel,
                                   (1, self.filter_size, num_classes, 1))

        self.kernel_x_comp = tf.ones_like(self.kernel_y)
        self.kernel_y_comp = tf.ones_like(self.kernel_x)

        self.kernel_x_sqr = self.kernel_x ** 2
        self.kernel_y_sqr = self.kernel_y ** 2

    def get_config(self):
        config = super().get_config()
        config['neighborhood_size'] = self.neighborhood_size
        return config

    def call(self, edges):
        tf.print(edges.shape)
        offset_x = tf.nn.depthwise_conv2d(edges,
                                          self.kernel_x,
                                          [1, 1, 1, 1],
                                          'SAME')
        offset_x = tf.nn.depthwise_conv2d(offset_x,
                                          self.kernel_x_comp,
                                          (1, 1, 1, 1),
                                          'SAME')

        offset_y = tf.nn.depthwise_conv2d(edges,
                                          self.kernel_y,
                                          (1, 1, 1, 1),
                                          'SAME')
        offset_y = tf.nn.depthwise_conv2d(offset_y,
                                          self.kernel_y_comp,
                                          (1, 1, 1, 1),
                                          'SAME')

        offset_norm = tf.sqrt(offset_x ** 2 + offset_y ** 2)

        sqr_x = tf.nn.depthwise_conv2d(edges,
                                       self.kernel_x_sqr,
                                       (1, 1, 1, 1),
                                       'SAME')
        sqr_x = tf.nn.depthwise_conv2d(sqr_x,
                                       self.kernel_x_comp,
                                       (1, 1, 1, 1),
                                       'SAME')

        sqr_y = tf.nn.depthwise_conv2d(edges,
                                       self.kernel_y_sqr,
                                       (1, 1, 1, 1),
                                       'SAME')
        sqr_y = tf.nn.depthwise_conv2d(sqr_y,
                                       self.kernel_y_comp,
                                       (1, 1, 1, 1),
                                       'SAME')
        norm_sqr = sqr_x + sqr_y

        return 2 * offset_norm / (norm_sqr + 1e-5)


if __name__ == "__main__":
    import numpy as np

    probs = np.zeros([1, 50, 50, 3], np.float32)
    probs[0, 10:20, 10:20, 0] = 1.
    probs[0, 25:30, 25:30, 1] = 1.
    probs[..., 2] = 1. - probs[..., 0] - probs[..., 1]

    # test soft_dist_centroid
    # sdc = soft_dist_centroid(probs)
    # sdc = np.array(sdc)
    # assert sdc.shape == (1, 3, 2)
    # assert sdc[0, 2, 0] > sdc[0, 0, 0] > sdc[0, 1, 0]
    #
    # print(soft_size(probs))

    from skimage.draw import disk
    import matplotlib.pyplot as plt
    probs = np.zeros([1, 50, 50, 2], np.float32)
    ind_x, ind_y = disk((25, 25), 10, shape=(50, 50))
    probs[0, ind_x, ind_y, 0] = 1.
    probs[..., 1] = 1. - probs[..., 0]

    curv_est = CurvatureEstimator(2, 3)
    grad = gradient2d(probs, tf.zeros([2, 2, 2]))
    curv = curv_est(grad)
    curv = np.array(curv)
    grad = np.array(grad)
    plt.imshow(grad[0, :, :, 0])
    plt.show()
    print(np.sum(curv, (1, 2)) / np.sum(grad, (1, 2)))
