#!/usr/bin/python3
import numpy as np
import cv2
import sys
import masks
import detect
import os
import glob


class work_env:
    
    def __init__(self, calib_stat, dir_path, data_path, video_fps):
        self.homos = calib_stat
        self.number_views = len(calib_stat)
        self.dir_path = dir_path
        self.data_path = data_path
        self.video_fps = video_fps
        self.frame_interval = 1000 / video_fps
        self.homo_invs, self.transforms =  masks.preprocess(self.homos)
        self.video_filepath = []
        self.frames = []
        self.output_frames = []


    '''preprocessing: frame the video into frames'''
    def get_frames(self):
        # fetch next frame
        if (len(self.frames) != 0):
            return (self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH), self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT), len(self.frames))
        
        # TODO: 
        # 1, main function swith off
        # 2, detect input change, gbr to gray single layer
        # 3, PIL save function embedded and show the result
        # 4, test detect and the other separately
        files = glob.glob(self.data_path + '*')
        if len(files) == 0: 
            raise RuntimeError("no data in directory: {0}".format(data_path))
        for filepath in files:
            if filepath.endswith("avi"):
                if not os.path.basename(filepath).startswith("output"):
                    self.video_filepath.append(filepath)
        
        print("video path: {0}".format(str(self.video_filepath)))

        fourcc = cv2.VideoWriter_fourcc(*'XVID') # avi codec
        for video in self.video_filepath:
            # get the frame
            cap = cv2.VideoCapture(video)
            height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # filename, fourcc-four char code of codec, fps, frameSize, [, isColor] 
            out = cv2.VideoWriter(os.path.join(self.data_path, 'output-' + os.path.basename(video)), fourcc, 20, (width, height))
            
            # add to array of frames
            self.output_frames.append(out)
            self.frames.append(cap)
        return (int(self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT)), len(self.frames))

    '''get the priority of different view by compare the overlapping area size'''
    def get_priorities(self):
        # manual setting, use view 0 for full tx
        manual_set_priorities = [0, 1, 2, 3]
        return manual_set_priorities

        # TODO strategy with overlap area calculation
    
    '''release videoCapure object'''
    def release_all(self):
        for idx in range(len(self.frames)):
            self.frames[idx].release()
            self.output_frames[idx].release()
        
def main():
    print("program launched with {0} arguments".format(len(sys.argv) - 1))
   
    # calibration input for four views, TODO replace stat with outsource
    calib_stat = masks.stat()
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(curr_dir_path, "../data/")
    video_fps = 40
    ENV = work_env(calib_stat, curr_dir_path, video_path, video_fps)

    # pipeline for each frame
    priorities = ENV.get_priorities()
    transforms = ENV.transforms
    homos = ENV.homos
    homo_invs = ENV.homo_invs

    print(np.shape(homos), np.shape(transforms))
    # print(priorities, homo_invs, transforms)
    
    print("get frame from videos")
    stats = ENV.get_frames()
    width = stats[0]
    height = stats[1]
    n = stats[2] # n is the number of views, usually 4 for our dataset
    print("val: {0} {1} {2}".format(width, height, n))

    frames = ENV.frames
    output_frames = ENV.output_frames
    ENV.release_all()

    # calculate the static masks
    print("start create overlap mask for frames, shape of transforms={0}, shape of priorities={1}".format(np.shape(transforms), np.shape(priorities)))
    stitch_masks = masks.create_img_mask(height, width, n, transforms, priorities)
    print('stitch masks: ', np.shape(stitch_masks))
    print(stitch_masks)

    # while(frames[0].isOpened()):
    while(1):
        ret = False
        input_images = []
        for idx in range(n):
            ret, frame = frames[idx].read()
            if frame is None or ret==True:
                print("frame lost for {0}th video, exit from the main loop".format(idx))
                return 
            input_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # TODO
        # get segmented image for non-pro and left-over images for non-pro
        # transform left-overs for each non-pro imgs
        # layerup segmented images onto the stiched imgs
        # multiprocess
        
        # overlap input images
        overlapped_input_images = []
        for idx in range(n):
            if index == priorities[0]:
                overlapped_input_images.append(None)
                continue
            overlapped_input_images.append(cv2.bitwise_and(src1=input_images[idx], src2=255 - stitch_masks[idx]).astype('uint8'))
        _, segmented_masks = detect.do_main(data=overlapped_input_images, priorities=priorities)
        print('segment mask: ', np.shape(segmented_masks))

        reduced_input_images = []
        for idx in range(n):
            if index == priorities[0]: 
                reduced_input_images = input_images[idx]
                continue
            # reverse mask
            reduced_input_images[idx] = cv2.bitwise_and(src1=input_images[idx], src2=(255 - segmented_masks[idx])).astype('uint8')    
        print('reduced input images: ', np.shape(recuded_input_images))
        # start video manipulation, apply to each frame 
        output_images = masks.transform_n_crop(reduced_input_images, priorities, stitch_masks, homos, homo_invs)
        output_images = overlap_images(output_images, priorities, segmented_masks, input_images) # overlap the segmented pic onto the transformed image

        for idx in range(n):
            output_frames[idx].write(output_images[idx])
        
        if ret == False: 
            break
        if cv2.waitKey(ENV.frame_interval) & 0xFF == ord('q'):
            break    

        # TODO removed when single frame works
        break
    
    # remove all 
    ENV.release_all()
   
if __name__ == '__main__':
    main() 
