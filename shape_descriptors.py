import tensorflow as tf
from keras.layers import Layer
from scipy.special import eval_hermitenorm


@tf.function
def soft_size(probabilities: tf.Tensor):
    rank = len(tf.shape(probabilities))
    return tf.reduce_sum(probabilities, tuple(range(1, rank - 1)))


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
    dilation = tf.nn.dilation2d(
        image, st_element, (1, 1, 1, 1), "SAME", "NHWC", (1, 1, 1, 1)
    )
    erosion = tf.nn.erosion2d(
        image, st_element, (1, 1, 1, 1), "SAME", "NHWC", (1, 1, 1, 1)
    )
    return dilation - erosion


# TODO: implement multiscale version
class CurvatureEstimator(Layer):
    """
    Estimator of the curvature.
    Based on the paper
        Scale Space Edge Curvature Estimation and
            Its Application to Straight Lines Detection
        Ekaterina V. Semeikina, Dmitry V. Yurin
    """

    def __init__(self, num_classes, sigma, filter_width=None, **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma
        self.filter_width = filter_width
        self.num_classes = num_classes

        self.sigma = sigma
        if filter_width is None:
            filter_width = int(4 * sigma)
        self.width = filter_width

        x = np.arange(-filter_width, filter_width + 1, dtype=np.float32)
        g0 = np.exp(-(x**2) / (2 * sigma**2))
        g0 /= np.sqrt(2 * np.pi * sigma)

        g = [g0]
        for n in range(1, 5):
            tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
            tmp /= sigma**n
            g.append(tmp)

        self.gx = [p.reshape([-1, 1, 1, 1]) for p in g]
        self.gy = [p.reshape([1, -1, 1, 1]) for p in g]

    def build(self, input_shape):
        num_channels = input_shape[0][-1]
        self.gx = [np.concatenate(num_channels * [p], 2) for p in self.gx]
        self.gy = [np.concatenate(num_channels * [p], 2) for p in self.gy]
        self.gx = [tf.constant(p, dtype=tf.float32) for p in self.gx]
        self.gy = [tf.constant(p, dtype=tf.float32) for p in self.gy]

        super().build(input_shape)

    def _get_derivatives(self, image):
        out = list()
        for n in [1, 3, 4]:
            tmp_out = list()
            for i in range(0, n + 1):
                tmp = tf.nn.depthwise_conv2d(
                    image, self.gx[n - i], (1, 1, 1, 1), "SAME"
                )
                tmp = tf.nn.depthwise_conv2d(
                    tmp, self.gy[i], (1, 1, 1, 1), "SAME")
                tmp_out.append(tmp)
            out.append(tmp_out)
        return out

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["filter_width"] = self.filter_width
        return config

    def call(self, inputs):
        image, edges = inputs
        derivs1, derivs3, derivs4 = self._get_derivatives(image)
        ux, uy = derivs1
        uxxx, uxxy, uxyy, uyyy = derivs3
        uxxxx, uxxxy, uxxyy, uxyyy, uyyyy = derivs4
        A = (uxxxx + uyyyy) * (ux**2) * (uy**2)
        A += uxxyy * (ux**4 + uy**4 - 4 * (ux**2) * (uy**2))
        A += 2 * ux * uy * (uy**2 - ux**2) * (uxxxy - uxyyy)

        B = tf.sqrt(ux**2 + uy**2)
        B2 = uxxx * (ux**3) + uyyy * (uy**3)
        B2 += 3 * ux * uy * (uxxy * ux + uxyy * uy)
        B = B * B2

        A = A * edges
        B = B * edges + (1.0 - edges) * 1e-10

        curv = -A / B

        return tf.abs(curv)


if __name__ == "__main__":
    import numpy as np

    probs = np.zeros([1, 50, 50, 3], np.float32)
    probs[0, 10:20, 10:20, 0] = 1.0
    probs[0, 25:30, 25:30, 1] = 1.0
    probs[..., 2] = 1.0 - probs[..., 0] - probs[..., 1]

    # test soft_dist_centroid
    # sdc = soft_dist_centroid(probs)
    # sdc = np.array(sdc)
    # assert sdc.shape == (1, 3, 2)
    # assert sdc[0, 2, 0] > sdc[0, 0, 0] > sdc[0, 1, 0]
    #
    # print(soft_size(probs))

    print("testing curvature")
    import matplotlib.pyplot as plt
    from skimage.draw import disk

    size = 151
    curv_est = CurvatureEstimator(2, 3.0)
    for r in [10, 20, 30, 40, 50, 60]:
        print(f"radius = {r}, 1/radius = {1 / r}")
        probs = np.zeros([1, size, size, 2], np.float32)
        ind_x, ind_y = disk((size // 2, size // 2), r, shape=(size, size))
        probs[0, ind_x, ind_y, 0] = 1.0
        probs[..., 1] = 1.0 - probs[..., 0]
        grad = gradient2d(probs, tf.zeros([3, 3, 2]))

        curv = curv_est([probs, grad])
        curv = np.array(curv)
        grad = np.array(grad)
        plt.imshow(curv[0, :, :, 0])
        plt.show()
        print(np.sum(curv, (1, 2)) / np.sum(grad, (1, 2)))
