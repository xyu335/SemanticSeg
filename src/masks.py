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


''' create the masks for all pairs of the views'''
''' now only support the fixed pairs: 0 - 1, 0 - 2, 0-3, 0-4'''
def create_img_mask(r, c, n, transforms, priorities):
    pro_idx = priorities[0]  # [0, 1, 2, 3]
    stitch_masks = np.array([pow(2, i) * np.ones((r, c)) for i in range(n)])
    return_masks = []
    corners = np.array([[0, 0],
                        [0, r],
                        [c, r],
                        [c, 0]]).astype(np.float)
    all_corners = corners
    for index in range(n):
        # create mask for index-th image

        if index == pro_idx:
            return_masks.append(None)
            continue
        else:
            warped_corners = transforms[index](corners)
            al_corners = np.vstack((all_corners, warped_corners))

            corner_min = np.min(al_corners, axis=0)
            corner_max = np.max(al_corners, axis=0)
            output_shape = (corner_max - corner_min)
            output_shape = np.ceil(output_shape[::-1])
            offset = SimilarityTransform(translation=-corner_min)
            offset_inv = SimilarityTransform(translation=corner_min)

            total_masks = []
            total_masks.append(warp(stitch_masks[index, :, :], (transforms[pro_idx] + offset).inverse, output_shape=output_shape, cval=0))
            total_masks.append(warp(stitch_masks[index, :, :], (transforms[index] + offset).inverse, output_shape=output_shape, cval=0))
            total_masks = np.sum(np.array(total_masks), axis=0)

            # return val
            transform_inv = ProjectiveTransform(transforms[index]._inv_matrix)
            return_masks.append(warp(total_masks, (offset_inv + transform_inv).inverse, output_shape=[r, c], cval=0))
            return_masks[index][(return_masks[index] % 1.0 != 0)] = pow(2, 1)  # pow(2,i)
            print(return_masks[index])
            
            ret_masks = return_masks[index]
            # now the image that has to be bitwise-and. so the background image has to be 255
            # the mask[i]. the overlap_label will be 2^len - 1 = 3, overlap label
            overlap_value = pow(2, 2) - 1 # 3
            ret_masks[(ret_masks != overlap_value)] = 255
            ret_masks[(ret_masks == overlap_value)] = 0 # reverse mask
            ret_masks = ret_masks.astype('uint8')
            print((ret_masks[index] == 255).sum())
    return return_masks


'''images array, priority array'''
def transform_n_crop(images, priorities, stitch_masks, homo, inv_homo):
    pro_idx = priorities[0]  # [2,1,3,0]
    recovered_imgs = []

    for i in range(len(images)):
        if i == pro_idx:
            recovered.append((images[i]).astype('uint8'))
            continue
        else:
            # transform the pro image and crop into the target image
            transform_pro_to_target = skimage.transform.ProjectiveTransform(np.matmul(inv_homo[i], homo[pro_idx]))  # inv_h1, h0
            trans_img_pro_to_target = warp(images[pro_idx], (transform_pro_to_target).inverse, cval=0)
            result = (images[i] + cv2.bitwise_and(src1=trans_img_pro_to_target.astye('uint8'), src2=(255 - stitch_masks[i]))).astype('uint8')  # masks[1]
            # maskedImg = Image.fromarray(result.astype('uint8'))
            recovered_imgs.append(result)

    return recovered_imgs


'''preprocess for the inv_homo mat and transform mat'''
def preprocess(homo):
    inv_homos = []
    transforms = []
    idm = np.eye(3)
    for i in range(np.shape(homo)[0]):
        inv_homos.append(np.linalg.inv(homo[i]))
        transforms.append(ProjectiveTransform(np.matmul(idm, homo[i])))
    return inv_homos, transforms


'''overlap segmented source images onto the transformed images
    note: segmented source image + transformed image together as source input from remote camera
'''
def overlap_images(trans_images, priorities, segmented_masks, src_images):
    output_images = []
    for idx in range(len(trans_images)):
        if idx == priorities[0]:
            output_images.append(src_images[idx]) 
            continue
        output = trans_images[idx]
        src_img = src_images[idx]
        R, C = np.shape(segmented_masks[0])
        for row in range(R):
            for col in range(C):
                if segmented_masks[row][col] == 255:
                    output[row][col] = src_img[row][col]
        output_images.append(output)
    return output_images


def stat():
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
    homos = [h0, h1, h2, h3]
    return homos

# im_0 = np.asarray(Image.open('0.jpg').convert("L"), dtype = np.float)
