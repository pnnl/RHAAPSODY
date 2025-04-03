
import cv2

from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers  import Flatten, Dense
from tensorflow.keras.models import Model

from scipy.spatial import distance

from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.restoration import denoise_wavelet
from skimage.transform import rescale


"""Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
"""


def denoise(image: np.ndarray):
    """Denoises and equalizes image

    Args:
      image: numpy array of ints, uints or floats
        Input data to be denoised. Image can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    Returns:
      ndarray of denoised image.
    """
    image = equalize_hist(denoise_wavelet(image))
    # artificially create third dimension
    return np.stack([image, image, image], axis=-1)


denoise_vec = np.vectorize(denoise, signature="(n)->(n)")


def create_embedding_model(n=None, m=None):
    """Return the vgg16 embedding model."""
    # vgg16 model
    vgg16 = tf.keras.applications.vgg16.VGG16(
        weights="imagenet", include_top=False, pooling="max", input_shape=(n, m, 3)
    )
    flatten = Flatten()
    new_layer2 = Dense(10, activation="softmax", name="my_dense_2")

    _in = vgg16.input  # vgg16.input
    _out = new_layer2(flatten(vgg16.output))  # vgg16.get_layer('fc2').output

    basemodel = Model(inputs=_in, outputs=vgg16.output)
    return basemodel


m = create_embedding_model()


class EmbeddingModel:
    """A class to handle the embedding model."""

    def __init__(self, image_size=None):
        if image_size is not None:
            self.update_image_size(image_size)
        else:
            self.image_size = None

    def update_image_size(self, new_image_size):
        """Update the model's image size."""
        self.image_size = new_image_size
        self.m = create_embedding_model(*self.image_size)

    def array_to_embedding(self, image: np.ndarray) -> np.ndarray:
        """Convert an image array to an embedding."""
        denoised_image = denoise(rescale(image, 0.25))
        if self.image_size is None:
            self.update_image_size((denoised_image.shape[0], denoised_image.shape[1]))
        return self.m.predict(denoised_image[np.newaxis, :, :, :], verbose=0)

    def tiff_to_array(self, img) -> np.ndarray:
        """Convert a tiff image to an image array."""
        # img = cv2.imread(tiff_path)

        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img1

    def tiff_to_embedding(self, img) -> np.ndarray:
        """Return the embedding of the image in the given tiff file."""
        image_array = img  # self.tiff_to_array(img)
        return self.array_to_embedding(image_array)
