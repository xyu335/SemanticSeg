#!/usr/bin/python3
import numpy as np
import cv2
import sys
import masks
import detect
import os


class work_env:
    
    def __init__(self, calib_stat, data_path, video_fps):
        self.homos = calib_stat
        self.number_views = len(calib)
        self.data_path = data_path
        self.video_fps = video_fps
        self.frame_interval = 1000 / video_fps
        self.homo_invs, self.transforms =  self.preprocess(self.homos)
        self.video_filepath = []
        self.frames = []
        self.output_frames = []


    '''preprocessing: frame the video into frames'''
    def get_frames(self):
        # fetch next frame
        if (len(self.frames) != 0):
            return [self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH), self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT, len(self.frames)]
        files = glob.glob(data_path)
        if len(files) == 0: 
            raise RuntimeError("no data in directory: {0}".format(data_path))
        for filepath in files:
            if filepath.endswith(".avi"):
                self.video_filepath.append(filepath)
    
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # avi codec
        for video in self.video_filepath:
            # get the frame
            cap = cv2.VideoCapture(video)
            height = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = cap = get(cv2.CAP_PROP_FRAME_HEIGHT)
            # filename, fourcc-four char code of codec, fps, frameSize, [, isColor] 
            out = cv2.VideoWriter(os.path.join(video.dirname, 'output-' + os.path.filename(video)), fourcc, 20.0, (width, height))
            
            self.output_frames.append(out)
            self.frames.append(cap)
        return [self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH), self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT, len(self.frames)]

    
    '''get the priority of different view by compare the overlapping area size'''
    def get_priorities(self):
        # manual setting, use view 0 for full tx
        manual_set_priorities = [0, 1, 2, 3]
        return manual_set_priorities

        # TODO strategy with overlap area calculation
        
def main():
    print("program launched with {0} arguments".format(len(sys.argv) - 1))
    
    # video path
    # calibration input for four views
    # calculate priorities
    calib_stat = masks.stat()
    curr_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(curr_path, "../data/")
    video_fps = 40
    ENV = work_env(calib_stat, curr_dir_path, video_path, video_fps)

    # build up the pipeline for each frame
    priorities = ENV.get_priorities()
    transforms = ENV.transforms
    homos = ENV.homos
    homo_invs = ENV.homo_invs

    stats = ENV.get_frames()
    width = stats[0]
    height = stats[1]
    n = stats[2]
    frames = ENV.frames
    output_frames = ENV.output_frames
    
    # calculate the static masks 
    masks = masks.create_img_mask(height, width, n, transforms, priorities)
    while(frames[0].isOpened()):
        ret = False
        input_images = []
        for idx in range(n):
            ret, frame = frames[idx].read()
            if frame0 is None or ret==True:
                print("frame lost for {0}th video".format(idx))
                break
            input_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # TODO
        # get segmented image for non-pro and left-over images for non-pro
        # transform left-overs for each non-pro imgs
        # layerup segmented images onto the stiched imgs
        # multiprocess
        segmented_masks = do_main(input_images)
        reduced_input_images = []
        for idx in range(n):
            if index == priorities[0]: 
                reduced_input_images = input_images[idx]
                continue
            reduced_input_images[idx] = cv2.bitwise_and(src1=input_images[idx], src2=(255 - segmented_masks[idx])).astype('uint8'),  
            

        # start video manipulation
        # apply to each frame 
        # concatenate each frame for a full video 
        output_images = masks.transform_n_crop(reduced_input_images, priorities, masks, homos, homo_invs)
        output_images = overlap_images(output_images, segmented_asks, input_images) # overlap the segmented pic onto the transformed image

        for idx in range(n):
            output_frames[idx].write(output_images[idx])
        
        if ret == False: 
            break
        if cv2.waitKey(ENV.frame_interval) & 0xFF == ord('q'):
            break    
    
    for idx in range(n):
        frames[idx].release()
        output_frames.release()
    
    
if __name__ == '__main__':
    main() 
