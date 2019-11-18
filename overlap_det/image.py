import cv2
import numpy as np 
from numpy.linalg import inv
from scipy.spatial import distance
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image
import skimage
from skimage.transform import SimilarityTransform, warp


def warp_images(image, transform, translation=None):
	r, c = image.shape[:2]
	# Note that transformations take coordinates in (x, y) format,
	# not (row, column), in order to be consistent with most literature
	corners = np.array([[0, 0],
						[0, r],
						[c, r],
						[c, 0]]).astype(np.float)

	# Warp the image corners to their new positions
	if translation is not None: 
		corners -= translation
	warped_corners = transform(corners)

	# Find the extents of both the reference image and the warped
	# target image
	all_corners = np.vstack((warped_corners))

	corner_min = np.min(all_corners, axis=0)
	corner_max = np.max(all_corners, axis=0)

	output_shape = (corner_max - corner_min)
	output_shape = np.ceil(output_shape[::-1])

	translation_0 = np.array([0.0, 0.0])
	if translation is not None: 
		translation_0 += translation

	offset_0 = SimilarityTransform(translation=-translation_0)
	offset = SimilarityTransform(translation=-corner_min)

	image = warp(image, (offset_0 + transform + offset).inverse, output_shape=output_shape, cval=0)

	im = Image.fromarray((image).astype('uint8'))


	return im, offset.translation



def get_inplace_images_conners(image, offset):
	print (offset)
	r, c = image.shape[:2]
	# Note that transformations take coordinates in (x, y) format,
	# not (row, column), in order to be consistent with most literature
	corners = np.array([[0, 0],
						[0, r],
						[c, 0],
						[c, r]]).astype(np.float)

	translated_corners = corners + offset

	# Find the extents of both the reference image and the warped
	# target image
	all_corners = np.vstack((corners, translated_corners))

	corner_min = np.min(all_corners, axis=0)
	corner_max = np.max(all_corners, axis=0)

	return corner_min, corner_max


def merge_two_images(image0, offset0, image1, offset1):
	cmin_0, cmax_0 = get_inplace_images_conners(image0, offset0)
	cmin_1, cmax_1 = get_inplace_images_conners(image1, offset1)

	out_min = np.min(np.vstack((cmin_0, cmin_1, cmax_0, cmax_1)), axis=0)
	out_max = np.max(np.vstack((cmin_0, cmin_1, cmax_0, cmax_1)), axis=0)

	print (out_min, out_max)

	output_shape = (out_max - out_min)
	output_shape = np.ceil(output_shape[::-1])

	im0_translation = SimilarityTransform(translation=-offset_0)
	im1_translation = SimilarityTransform(translation=-offset_1)
	all_translation = SimilarityTransform(translation=-out_min)

	image0 = warp(image0, (im0_translation+all_translation).inverse, output_shape=output_shape, cval=0)
	image1 = warp(image1, (im1_translation+all_translation).inverse, output_shape=output_shape, cval=0)

	im0 = Image.fromarray((image0).astype('uint8'))
	im1 = Image.fromarray((image1).astype('uint8'))

	im0.show()
	im1.show()


 
if __name__ == '__main__' :

	h0 = np.array([[0.176138,    0.647589,   -63.412272],
				   [-0.180912,   0.622446,   -0.125533],
				   [-0.000002,   0.001756,   0.102316]])

	h1 = np.array([[0.177291,    0.004724,   31.224545],
				   [0.169895,    0.661935,   -79.781865],
				   [-0.000028,   0.001888,   0.054634]])

	h2 = np.array([[-0.118791,	 0.077787,	 64.819189],
				   [0.133127,	 0.069884,	 15.832922],
				   [-0.000001,	 0.002045,	 -0.057759]])

	h3 = np.array([[-0.142865,	 0.553150,	 -17.395045],
				   [-0.125726,	 0.039770,	 75.937144],
				   [-0.000011,	 0.001780,	 0.015675]])
 

	im_0 = np.asarray(Image.open('0.jpg').convert("L"), dtype = np.float)
	im_1 = np.asarray(Image.open('1.jpg').convert("L"), dtype = np.float)	
	im_2 = np.asarray(Image.open('2.jpg').convert("L"), dtype = np.float)
	im_3 = np.asarray(Image.open('3.jpg').convert("L"), dtype = np.float)

	inv_h0 = np.linalg.inv(h0)
	inv_h1 = np.linalg.inv(h1)
	inv_h2 = np.linalg.inv(h2)
	inv_h3 = np.linalg.inv(h3)

	transform0 = skimage.transform.ProjectiveTransform(h0)
	transform1 = skimage.transform.ProjectiveTransform(h1)
	transform2 = skimage.transform.ProjectiveTransform(h2)
	transform3 = skimage.transform.ProjectiveTransform(h3)


	transform01 = skimage.transform.ProjectiveTransform(np.matmul(inv_h0, h1))

	im_0_trans, offset_0 = warp_images(im_0, transform0)
	im_1_trans, offset_1 = warp_images(im_1, transform1)
	im_2_trans, offset_2 = warp_images(im_2, transform2)
	im_3_trans, offset_3 = warp_images(im_3, transform3)

	im_01_trans, offset_01 = warp_images(im_1, transform01)

	im_0_trans.show()
	im_1_trans.show()
	im_01_trans.show()

	# im_2_trans.show()
	# im_3_trans.show()

	# im_0_trans.save("0_trans.png")
	# im_1_trans.save("1_trans.png")

	# im0_naiv = warp(im_0, transform0.inverse)
	# im1_naiv = warp(im_1, transform1.inverse)

	# im0_naiv = Image.fromarray((im0_naiv).astype('uint8'))
	# im1_naiv = Image.fromarray((im1_naiv).astype('uint8'))

	# im0_naiv.save("0_naiv.png")
	# im1_naiv.save("1_naiv.png")

	
	# im_0_trans = np.asarray(im_0_trans, dtype = np.float)
	# im_1_trans = np.asarray(im_1_trans, dtype = np.float)

	# rev_transform0 = skimage.transform.ProjectiveTransform(np.linalg.inv(h0))
	# rev_transform1 = skimage.transform.ProjectiveTransform(np.linalg.inv(h1))
 
	# im_0_recover, _ = warp_images(im_0_trans, rev_transform0, offset_0)
	# im_1_recover, _ = warp_images(im_1_trans, rev_transform1, offset_1)


	im_01_trans = np.asarray(im_1_trans, dtype = np.float)

	rev_transform01 = skimage.transform.ProjectiveTransform(transform01._inv_matrix)

	im_01_recover, _ = warp_images(im_01_trans, rev_transform01, offset_01)

	im_01_recover.show()

	# im_0_recover.show()
	# im_1_recover.show()

	
	# im_0_trans = np.asarray(im_0_trans, dtype = np.float)
	# im_1_trans = np.asarray(im_1_trans, dtype = np.float)
	# merge_two_images(im_0_trans, offset_0, im_1_trans, offset_1)


