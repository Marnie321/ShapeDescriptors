import tensorflow as tf
from keras.layers import Layer
from scipy.special import eval_hermitenorm


# TODO: implement multiscale version


class CurvatureEstimator(Layer):
    """
    Estimator of the curvature.
    This one estimates the curvature for a single parameter sigma.

    Based on the paper
        Scale Space Edge Curvature Estimation and
            Its Application to Straight Lines Detection
        Ekaterina V. Semeikina, Dmitry V. Yurin
    """

    def __init__(
        self,
        sigma,
        filter_width=None,
        st_element_size=3,
        edges_to_check=None,
        **kwargs
    ):
        """
        Arguments:
           sigma: scale parameter of the Gaussian derivatives
        Optional keyword arguments:
            filter_width: size of the Gaussian derivatives.
                          int(4 * sigma) if not specified.
            st_element_size: size of the structuring element to compute edges.
                             Defaults to 3
            edges_to_check: None or tuple of tuples of two integers.

        """
        super().__init__(**kwargs)

        self.num_channels = None
        self.st_element = None
        self.sigma = sigma
        self.filter_width = filter_width
        self.st_element_size = st_element_size
        self.edges_to_check = edges_to_check

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
        num_channels = input_shape[-1]
        if self.edges_to_check is None:
            self.edges_to_check = (
                (i, j) for i in range(num_channels)
                for j in range(i + 1, num_channels)
            )

        self.num_channels = num_channels
        self.st_element = tf.zeros(
            [self.st_element_size, self.st_element_size,
                self.num_channels], tf.float32
        )
        self.gx = [np.concatenate(num_channels * [p], 2) for p in self.gx]
        self.gy = [np.concatenate(num_channels * [p], 2) for p in self.gy]
        self.gx = [tf.constant(p, dtype=tf.float32) for p in self.gx]
        self.gy = [tf.constant(p, dtype=tf.float32) for p in self.gy]

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["filter_width"] = self.filter_width
        config["st_element_size"] = self.st_element_size
        config["edges_to_check"] = self.edges_to_check
        return config

    def _get_edges(self, image):
        dilated = tf.nn.dilation2d(
            image, self.st_element, (1, 1, 1, 1), "SAME", "NHWC", (1, 1, 1, 1)
        )
        edges = list()
        for i, j in self.edges_to_check:
            tmp = dilated[:, :, :, i] * dilated[:, :, :, j]
            edges.append(tmp)
        edges = tf.stack(edges, -1)
        return edges

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

    def _get_curvature(self, image, edge_max):
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

        curv = tf.where(tf.abs(edge_max) > 1e-6, -A / B, 0.0)

        return curv

    def _get_curv_per_pixel_edge(self, curv, edges):
        curv_per_edge = list()
        for n, (i, j) in enumerate(self.edges_to_check):
            edge = edges[:, :, :, n]
            edge_curv = 0.5 * (curv[:, :, :, i] - curv[:, :, :, j])
            edge_curv = edge * edge_curv
            curv_per_edge.append(edge_curv)
        curv_per_edge = tf.stack(curv_per_edge, -1)
        return curv_per_edge

    def call(self, image):
        edges = self._get_edges(image)
        edge_mask = tf.reduce_max(edges, -1, keepdims=True)
        curv = self._get_curvature(image, edge_mask)

        curv_per_edge = self._get_curv_per_pixel_edge(curv, edges)

        curv_per_edge_mean = tf.reduce_sum(curv_per_edge, [1, 2]) / \
            tf.reduce_sum(edges, [1, 2])

        return curv_per_edge_mean


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.draw import disk

    # NOTE:  For circles, the estimated curvature is approximately proportional
    #       to the actual curvature 1 / r when sigma = 3 (with a ratio of 1.6).
    #        Why is this?
    print("testing curvature")
    size = 151
    curv_est = CurvatureEstimator(2, 5.0, edges_to_check=[(0, 1)])
    for r in [10, 20, 30, 40, 50, 60]:
        print(f"radius = {r}, 1/radius = {1 / r}")
        probs = np.zeros([1, size, size, 2], np.float32)
        ind_x, ind_y = disk((size // 2, size // 2), r, shape=(size, size))
        probs[0, ind_x, ind_y, 0] = 1.0
        probs[..., 1] = 1.0 - probs[..., 0]
        probs = tf.constant(probs)

        curv = curv_est(probs)
        curv = np.array(curv)

        edges = curv_est._get_edges(probs)
        curv_image = curv_est._get_curvature(
            probs,
            tf.reduce_max(edges, -1, keepdims=True))
        curv_image = curv_est._get_curv_per_pixel_edge(curv_image, edges)
        curv_image = np.array(curv_image)
        plt.imshow(curv_image[0, :, :, 0])
        plt.show()
        print(curv)
        # plt.imshow(curv[0, :, :, 0])
        # plt.show()
