import numpy as np
import tensorflow as tf
from keras.layers import Layer
from skimage import draw


class CurvatureEstimator(Layer):
    """
    Estimator of the curvature based on the ratio of the area of a disk inside 
    the region and  the total area.

    Based on the paper
        Estimation of the curvature of an interface from a digital 2D image
            Olav Inge Frette, George Virnovsky, Dmitriy Silin
    """

    def __init__(
        self,
        disk_radius,
        edges_to_check=None,
        st_element_size=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        size = 2 * disk_radius + 1
        ind_x, ind_y = draw.disk(
            (size // 2, size // 2),
            disk_radius,
            shape=(size, size))
        disk = np.zeros([size, size], np.float32)
        disk[ind_x, ind_y] = 1.

        self.filter_size = size
        self.disk_radius = disk_radius
        self.num_channels = None
        self.disk_area = float(len(ind_x))
        self.disk = disk
        self.disk = tf.constant(self.disk)
        self.disk = tf.reshape(
            self.disk, (self.filter_size, self.filter_size, 1, 1))

        self.st_element_size = st_element_size
        self.st_element = None
        self.edges_to_check = edges_to_check

    def build(self, input_shape):
        num_channels = input_shape[-1]
        if self.edges_to_check is None:
            self.edges_to_check = (
                (i, j) for i in range(num_channels)
                for j in range(i + 1, num_channels)
            )

        self.num_channels = num_channels
        self.disk = tf.tile(self.disk, (1, 1, num_channels, 1))

        self.st_element = tf.zeros(
            [self.st_element_size, self.st_element_size,
                self.num_channels], tf.float32
        )

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config["disk_radius"] = self.disk_radius
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

    def _get_curvature(self, image):
        partial_area = tf.nn.depthwise_conv2d(
            image, self.disk, (1, 1, 1, 1), "SAME"
        )
        curv = partial_area / self.disk_area - 1 / 2
        curv *= 3 * np.pi / self.disk_radius
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
        curv = self._get_curvature(image)

        curv_per_edge = self._get_curv_per_pixel_edge(curv, edges)

        curv_per_edge_mean = tf.reduce_sum(curv_per_edge, [1, 2]) / \
            tf.reduce_sum(edges, [1, 2])

        return curv_per_edge_mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("testing curvature")
    size = 151
    curv_est = CurvatureEstimator(5, [(0, 1)])
    for r in [10, 20, 30, 40, 50, 60]:
        print(f"radius = {r}, 1/radius = {1 / r}")
        probs = np.zeros([1, size, size, 2], np.float32)
        ind_x, ind_y = draw.disk((size // 2, size // 2), r, shape=(size, size))
        probs[0, ind_x, ind_y, 0] = 1.0
        probs[..., 1] = 1.0 - probs[..., 0]
        probs = tf.constant(probs)

        curv = curv_est(probs)
        curv = np.array(curv)

        edges = curv_est._get_edges(probs)
        curv_image = curv_est._get_curvature(probs)
        curv_image = curv_est._get_curv_per_pixel_edge(curv_image, edges)
        curv_image = np.array(curv_image)
        plt.imshow(curv_image[0, :, :, 0])
        plt.show()
        print(curv)
        # plt.imshow(curv[0, :, :, 0])
        # plt.show()
