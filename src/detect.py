#!/usr/bin/python3
import torch
from torchvision.models import detection
from torchvision import transforms as T

import sys
import os 
import glob
from PIL import Image
import cv2
import numpy as np
import random

model = None

'''get_mask'''
def get_mask_gray(prediction, src_img, th_mask, th_score):
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels'] # used for detection other than human
    boxes = prediction[0]['boxes']
    label_person = 1
    print('shape of mask, scores, labels: {0}, {1},{2}'.format(masks.size(),scores.size(), labels.size()))
    
    alpha_shape = (np.shape(src_img)[0], np.shape(src_img)[1], 1)
    alpha = np.zeros((alpha_shape[0], alpha_shape[1]), np.uint8)
    print('H, W, C: ', alpha_shape)
    # tensor to ndarray
    masks = masks.detach().numpy().squeeze(1);
    # for save matrix value to file
    np.set_printoptions(threshold=sys.maxsize)

    #instance level
    for idx in range(boxes.shape[0]):
        if scores[idx] > th_score and labels[idx] == label_person:
            mask = masks[idx]
            count = 0
            for y in range(alpha_shape[0]):
                for x in range(alpha_shape[1]):
                    if mask[y][x] > th_mask:
                        alpha[y][x] = 255 # range in 0 to 255 for transp
                        count = count + 1
            print('hit once', ' count: ', count)
    return alpha

'''
    score_threshold:    default set to 0.75 to accept acceptable classfied objects
    mask threshold:     deafult set to be 0.5 for soft mask border
    class_num:          default set to be 91 categories 
'''
def do_main(data=None, output_path=None, priorities=None, th_mask=0.5, th_scores=0.75, class_num=91):
    src_img = None
    if data is not None and isinstance(data, np.ndarray):
        src_img = data
        if output_path is None:
            output_path = os.getcwd() + "temp_output.png"
    else: 
        raise RuntimeError("invalid data input") 

    if model is None:
        model = detection.maskrcnn_resnet50_fpn(num_classes=91, pretrained=True)
        model.eval()
    # in_features = model.roi_heads.box_predictor.cls_score.in_features 
    # model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, class_num)
    
    masked_img=None
    masks=None
    masks = []
    for idx in range(data[0]):
        if idx == priorities[0]:
            masks.append(None)
            continue
        input_img = []
        # prepare ndarray with normalization and switch channel
        img = torch.from_numpy(src_img/255.).permute(2,0,1).float()
        input_img.append(img)
        
        prediction = model(input_img)
        mask = apply_mask(prediction, priorities, src_img, th_mask, th_scores)
        masks.append(mask)
    return masks
if __name__ == '__main__':
    do_main()
    print("finish")
