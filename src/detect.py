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

def add_box(scores, labels, boxes, masks, src_img, output_path):
    alpha_shape = (np.shape(src_img)[0], np.shape(src_img)[1], 1)
    alpha = np.zeros((alpha_shape[0], alpha_shape[1]), np.uint8)
    print('H, W, C: ', alpha_shape)
    masks = masks.detach().numpy().squeeze(1);
    
    combined_mask = []
    collision = 0
    np.set_printoptions(threshold=sys.maxsize)
    for idx in range(boxes.shape[0]):
        if scores[idx] > 0.75 and labels[idx] == 1:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2],boxes[idx][3]
            cv2.rectangle(src_img, (x1, y1), (x2, y2), (100, 100, 100),thickness=2)
            combined_mask.append(masks[idx])
            mask = masks[idx]
            count = 0
            #with open('output' + str(idx) + '.txt', 'w') as f:
            #    f.write(np.array2string(mask))
            for y in range(alpha_shape[0]):
                for x in range(alpha_shape[1]):
                    if mask[y][x] > 0.5:
                        alpha[y][x] = 1
                        count = count + 1
                        # if (alpha[y][x] == 1):
                        # collision = collision + 1
            print('hit once', (x1,x2,y1,y2), ' count: ', count)
    
    #alpha = np.expand_dims(alpha, axis=2)
    print('shape of combined mask: ', np.shape(combined_mask))
    #img = np.concatenate((src_img, alpha), axis = 2)
    #trans_img = np.transpose(src_img, (2,0,1))
    b,g,r = cv2.split(src_img)
    # print('shape: ', src_img[0].shape, src_img[1].shape, src_img[2].shape,alpha.shape)
    with open('alpha.txt', 'w') as f:
        f.write(np.array2string(alpha))

    final_img = cv2.merge((b, g, r, alpha))
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path + '.png', final_img)

def do_main():
    # current_path = os.getcwd()
    current_path = os.path.dirname(os.path.realpath(__file__))
    files = glob.glob(os.path.join(current_path, "*"))
    model = detection.maskrcnn_resnet50_fpn(num_classes=91, pretrained=True)
    # number of classes got about 91 kinds
    # num_classes = 2
    # in_features = model.roi_heads.box_predictor.cls_score.in_features 
    # model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.eval()
    for filepath in files:
        if filepath.endswith(".png") and filepath.find('cam0') != -1:
            print("input filepath: " + filepath)
            filename = os.path.basename(filepath)
            output = 'output_' + filename[:-4] 
            output_filepath = output + '.txt'
            print("output filename: {0}".format(output_filepath))
            '''
            img = Image.open(filepath)
            img = img.convert('RGB')
            print(img.size)
            input_img = get_transform(False)(img)
            input_img = input_img.unsqueeze(0)
            print('tensor shape: {0}'.format(input_img.size()))
            '''
            input_img = []
            src_img = cv2.imread(filepath)
            img = torch.from_numpy(src_img/255.).permute(2,0,1).float()
            input_img.append(img)
            prediction = model(input_img)
            
            masks = prediction[0]['masks']
            scores = prediction[0]['scores']
            labels = prediction[0]['labels']
            boxes = prediction[0]['boxes']

            print('shape of m, s, l: {0}, {1},{2}'.format(masks.size(),scores.size(), labels.size()))
            print(scores)
            
            add_box(scores, labels, boxes, masks, src_img, output)
            #result = torch.as_tensor((masks > 0.5), dtype = torch.int32)
            #result.squeeze(1)
            #result = result.numpy()
            #with open(output_filepath, 'w') as f:
            #    np.set_printoptions(threshold=sys.maxsize)
            #    first_mask = result[0]
            #    f.write(np.array2string(first_mask))
            #torch.save(result, output_filepath)
            
            #get masked img
            #new_img = get_masked_image()
            #print(np.shape(new_img))


if __name__ == '__main__':
    do_main()
    print("finish")
