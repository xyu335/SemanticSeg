import numpy as np
from scipy.spatial import distance
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image
import skimage
from skimage.transform import SimilarityTransform, warp


def warp_images(image0, image1, transform):
    r, c = image1.shape[:2]
    # Note that transformations take coordinates in (x, y) format,
    # not (row, column), in order to be consistent with most literature
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions
    warped_corners = transform(corners)

    # Find the extents of both the reference image and the warped
    # target image
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse, output_shape=output_shape, cval=-1)

    image1_ = warp(image1, (transform + offset).inverse, output_shape=output_shape, cval=-1)

    image0_zeros = warp(image0, offset.inverse, output_shape=output_shape, cval=0)

    image1_zeros = warp(image1, (transform + offset).inverse, output_shape=output_shape, cval=0)

    overlap = (image0_ != -1.0 ).astype(int) + (image1_ != -1.0).astype(int)
    overlap += (overlap < 1).astype(int)
    merged = (image0_zeros+image1_zeros)/overlap

    im = Image.fromarray((merged).astype('uint8'))
    im.save('stitched_images.jpg')
    im.show()


h0 = np.array([[0.176138,    0.647589,   -63.412272],
               [-0.180912,   0.622446,   -0.125533],
               [-0.000002,   0.001756,   0.102316]])

h1 = np.array([[0.177291,    0.004724,   31.224545],
               [0.169895,    0.661935,   -79.781865],
               [-0.000028,   0.001888,   0.054634]])

h2 = np.array([[-0.118791,   0.077787,   64.819189],
               [0.133127,    0.069884,   15.832922],
               [-0.000001,   0.002045,   -0.057759]])

h3 = np.array([[-0.142865,   0.553150,   -17.395045],
               [-0.125726,   0.039770,   75.937144],
               [-0.000011,   0.001780,   0.015675]])

inv_h0 = np.linalg.inv(h0)
inv_h1 = np.linalg.inv(h1)
inv_h2 = np.linalg.inv(h2)
inv_h3 = np.linalg.inv(h3)

im_0 = np.asarray(Image.open('0.jpg').convert("L"), dtype = np.float)
im_1 = np.asarray(Image.open('1.jpg').convert("L"), dtype = np.float)

transform = skimage.transform.ProjectiveTransform(np.matmul(inv_h1, h0))
warp_images(im_1, im_0, transform)