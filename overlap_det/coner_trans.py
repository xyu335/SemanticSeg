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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def warp_images(image, transform):
	r, c = image.shape[:2]
	# Note that transformations take coordinates in (x, y) format,
	# not (row, column), in order to be consistent with most literature
	corners = np.array([[0, 0],
						[0, r],
						[c, r],
						[c, 0]]).astype(np.float)

	warped_corners = transform(corners)

	return warped_corners


 
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

 

	im_0 = np.asarray(Image.open('0.jpg').convert("L"), dtype = np.float)
	im_1 = np.asarray(Image.open('1.jpg').convert("L"), dtype = np.float)	
	im_2 = np.asarray(Image.open('2.jpg').convert("L"), dtype = np.float)
	im_3 = np.asarray(Image.open('3.jpg').convert("L"), dtype = np.float)

	transform0 = skimage.transform.ProjectiveTransform(np.matmul(inv_h0, h0))
	transform1 = skimage.transform.ProjectiveTransform(np.matmul(inv_h0, h1))
	transform2 = skimage.transform.ProjectiveTransform(np.matmul(inv_h0, h2))
	transform3 = skimage.transform.ProjectiveTransform(np.matmul(inv_h0, h3))

	coners_0 = warp_images(im_0, transform0)
	coners_1 = warp_images(im_1, transform1)
	coners_2 = warp_images(im_2, transform2)
	coners_3 = warp_images(im_3, transform3)

	print (coners_0)
	print (coners_1)
	print (coners_2)
	print (coners_3)

	fig, ax = plt.subplots()
	patches = []

	for shape in [coners_0, coners_1, coners_2, coners_3]:
		polygon = Polygon(shape, False)
		patches.append(polygon)

	p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

	colors = 100*np.random.rand(len(patches))
	p.set_array(np.array(colors))

	ax.add_collection(p)

	ax.autoscale_view()
	plt.show()