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
from skimage.transform import SimilarityTransform, ProjectiveTransform, warp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def create_image_mask(images, transforms):
	n, r, c = images.shape[:3]
	print (n, r, c)
	masks = np.array([pow(2, i) * np.ones((r,c)) for i in range(n)])

	corners = np.array([[0, 0],
						[0, r],
						[c, r],
						[c, 0]]).astype(np.float)

	all_corners = corners
	for i in range(n):
		warped_corners = transforms[i](corners)
		all_corners = np.vstack((all_corners, warped_corners))

	# print (all_corners)

	corner_min = np.min(all_corners, axis=0)
	corner_max = np.max(all_corners, axis=0)


	output_shape = (corner_max - corner_min)
	output_shape = np.ceil(output_shape[::-1])

	offset = SimilarityTransform(translation= -corner_min)
	offset_inv = SimilarityTransform(translation= corner_min)

	total_masks = []
	for i in range(n):
		total_masks.append(warp(masks[i,:,:], (transforms[i] + offset).inverse, output_shape=output_shape, cval=0))
	total_masks = np.sum(np.array(total_masks), axis=0)

	ret_masks = []
	for i in range(n):
		transform_inv = ProjectiveTransform(transforms[i]._inv_matrix)
		ret_masks.append(warp(total_masks, (offset_inv + transform_inv).inverse, output_shape=[r, c], cval=0))
		ret_masks[i][(ret_masks[i]%1.0 != 0)] = pow(2, i)
		print ((ret_masks[i] == 3.0).sum())

	return ret_masks

 
if __name__ == '__main__' :

	h0 = np.array([[0.176138, 0.647589,    -63.412272],
				  [-0.180912,   0.622446,    -0.125533],
				  [-0.000002,   0.001756,    0.102316]])

	h1 = np.array([[0.177291,    0.004724,    31.224545],
				   [0.169895,   0.661935,    -79.781865],
				   [-0.000028,    0.001888,    0.054634]])

	h2 = np.array([[-0.118791,	0.077787,	64.819189],
				   [0.133127,	0.069884,	15.832922],
				   [-0.000001,	0.002045,	-0.057759]])

	h3 = np.array([[-0.142865,	0.553150,	-17.395045],
				   [-0.125726,	0.039770,	75.937144],
				   [-0.000011,	0.001780,	0.015675]])

	inv_h0 = np.linalg.inv(h0)
	inv_h1 = np.linalg.inv(h1)
	inv_h2 = np.linalg.inv(h2)
	inv_h3 = np.linalg.inv(h3)
	idm = np.eye(3)

	im_0 = np.asarray(Image.open('0.jpg').convert("L"), dtype = np.float)
	im_1 = np.asarray(Image.open('1.jpg').convert("L"), dtype = np.float)	
	im_2 = np.asarray(Image.open('2.jpg').convert("L"), dtype = np.float)
	im_3 = np.asarray(Image.open('3.jpg').convert("L"), dtype = np.float)

	transform0 = skimage.transform.ProjectiveTransform(np.matmul(idm, h0))
	transform1 = skimage.transform.ProjectiveTransform(np.matmul(idm, h1))
	transform2 = skimage.transform.ProjectiveTransform(np.matmul(idm, h2))
	transform3 = skimage.transform.ProjectiveTransform(np.matmul(idm, h3))

	images = np.array([im_0, im_1])
	transforms = [transform0, transform1]

	masks = create_image_mask(images, transforms)

	for each in masks:
		print ((each == 3.0).sum())


	overlap_value = pow(2, len(images)) - 1
	overlap_area = [(masks[i]==overlap_value).sum() for i in range(len(masks))]
	pro_idx = overlap_area.index(max(overlap_area))


	masked_im_1 = None
	for i in range(len(images)):
		if i == pro_idx:
			maskedImg = Image.fromarray((images[i]).astype('uint8'))
			maskedImg.show()
			continue

		masks[i][(masks[i]!=overlap_value)] = 255
		masks[i][(masks[i]==overlap_value)] = 0
		masks[i] = masks[i].astype('uint8')

		maskedImg = cv2.bitwise_and(src1 = images[i].astype('uint8'), src2 = masks[i]).astype('uint8')
		masked_im_1 = maskedImg
		maskedImg = Image.fromarray(maskedImg.astype('uint8'))
		maskedImg.show()


	transform0_1 = skimage.transform.ProjectiveTransform(np.matmul(inv_h1, h0))
	trans_im_0_1 = warp(images[0], (transform0_1).inverse, cval=0)
	result = (masked_im_1 + cv2.bitwise_and(src1 = trans_im_0_1.astype('uint8'), src2 = (255 - masks[1]))).astype('uint8')
	maskedImg = Image.fromarray(result.astype('uint8'))
	maskedImg.show()

