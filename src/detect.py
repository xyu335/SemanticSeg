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

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if (train):
        transform.append(T.RandomHorizontalFlip(0.5))
        #transform.append(T.mead[])
        #transform.append(T.stddev[])
    return T.Compose(transforms) 

def random_color():
    b = random.randint(0, 255)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    return (r, g, b)

def get_masked_image(mask, img, threshold=None): 
    # C * H * W
    new_img = np.concatenate(img, mask, axis=0)
    return new_img

def apply_mask(prediction, src_img, th_mask, th_score, output_path,mode='mask_only'):
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels'] # used for detection other than human
    boxes = prediction[0]['boxes']
    label_person = 1
    box_detection = False
    if box_detection:
        boxes = prediction[0]['boxes'] # for box detection
    print('shape of mask, scores, labels: {0}, {1},{2}'.format(masks.size(),scores.size(), labels.size()))
    
    alpha_shape = (np.shape(src_img)[0], np.shape(src_img)[1], 1)
    alpha = np.zeros((alpha_shape[0], alpha_shape[1]), np.uint8)
    print('H, W, C: ', alpha_shape)
    # tensor to ndarray
    masks = masks.detach().numpy().squeeze(1);
    pixel_collision = 0
    # for save matrix value to file
    np.set_printoptions(threshold=sys.maxsize)
    for idx in range(boxes.shape[0]):
        if scores[idx] > th_score and labels[idx] == label_person:
            if box_detection:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2],boxes[idx][3]
                cv2.rectangle(src_img, (x1, y1), (x2, y2), (100, 100, 100),thickness=2)
            mask = masks[idx]
            count = 0
            for y in range(alpha_shape[0]):
                for x in range(alpha_shape[1]):
                    if mask[y][x] > th_mask:
                        alpha[y][x] = 255 # range in 0 to 255 for transp
                        count = count + 1
                        # if (alpha[y][x] == 1):
                        # collision = collision + 1
            # print('hit once', (x1,x2,y1,y2), ' count: ', count)
            print('hit once', ' count: ', count)
    if mode == 'mask_and_file' and src_img is not None and output_path is not None:
        b,g,r = cv2.split(src_img)
        final_img = cv2.merge((b, g, r, alpha))
        # final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        image_path = output_path + '.png'
        cv2.imwrite(image_path, final_img)
        print('image saved to {0}'.format(image_path))
        return (alpha, final_img)
    elif mode == 'mask_only': 
        print('mask saved, no image is saved')
        return (alpha, None)
    else: 
        raise RuntimeError('not supported')
        
        
'''
    score_threshold:    default set to 0.75 to accept acceptable classfied objects
    mask threshold:     deafult set to be 0.5 for soft mask border
    class_num:          default set to be 91 categories 
'''
def do_main(data=None, output_path=None, th_mask=0.5, th_scores=0.75, class_num=91):
    if data is not None and isinstance(data, np.ndarray):
        src_img = data
        if output_path is None:
            output_path = os.getcwd() + "temp_output.png"
    else: 
        current_dir = os.path.dirname(os.path.realpath(__file__))
        files = glob.glob(os.path.join(current_dir, "../data/*"))
        if len(files) == 0:
            return None
            # raise RuntimeError('There is no data for input')
        filelist = []
        for filepath in files:
            if filepath.endswith(".png") and filepath.find('cam') != -1:
                filelist.append(filepath)
                
    
    model = detection.maskrcnn_resnet50_fpn(num_classes=91, pretrained=True)
    model.eval()
    # in_features = model.roi_heads.box_predictor.cls_score.in_features 
    # model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, class_num)
    
    for filepath in filelist: 
        filename = os.path.basename(filepath)
        output_path = os.path.join(current_dir, '../output/output_') + filename[:-4] 
        src_img = cv2.imread(filepath)
        print("input filepath: {0}".format(filepath))
        print("output filename: {0}".format(output_path))
        
        input_img = []
        # prepare ndarray with normalization and switch channel
        img = torch.from_numpy(src_img/255.).permute(2,0,1).float()
        input_img.append(img)
        prediction = model(input_img)
        # print(scores)
        masked_img, masks = apply_mask(prediction, src_img, th_mask, th_scores,output_path, mode='mask_and_file')

if __name__ == '__main__':
    do_main()
    print("finish")
