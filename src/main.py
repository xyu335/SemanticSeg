#!/usr/bin/python3
import numpy as np
import cv2
import sys
import masks
import detect
import os
import glob
import time

class work_env:
    
    def __init__(self, calib_stat, dir_path, data_path, video_fps):
        self.homos = calib_stat
        self.number_views = len(calib_stat)
        self.dir_path = dir_path
        self.data_path = data_path
        self.video_fps = video_fps
        self.frame_interval = 1 # framing frequency
        self.homo_invs, self.transforms =  masks.preprocess(self.homos)
        self.video_filepath = []
        self.frames = []
        self.output_frames = []
        self.model = None
        self.video_pos = None
        self.produced_frame_count = 0

    '''preprocessing: frame the video into frames'''
    def get_frames(self, offset_frames, tagid):
        # fetch next frame
        if (len(self.frames) != 0):
            return (self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH), self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT), len(self.frames))
        
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
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # filename, fourcc-four char code of codec, fps, frameSize, [, isColor] 
            output_path = os.path.join(self.data_path, 'output-' + str(tagid) + '-' + os.path.basename(video))
            out = cv2.VideoWriter(output_path, fourcc, self.video_fps, (width, height), False)
            print("video path: {0}, capture openable: {1}, h & w: {2} {3}, output_path: {4}".format(video, cap.isOpened(), height, width, output_path))
            # add to array of frames
            self.output_frames.append(out)
            self.frames.append(cap)
        self.video_pos = [offset_frames] * len(self.frames)
        return (int(self.frames[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.frames[0].get(cv2.CAP_PROP_FRAME_HEIGHT)), len(self.frames))

    '''get the priority of different view by compare the overlapping area size'''
    def get_priorities(self):
        # manual setting, use view 0 for full tx
        manual_set_priorities = [0, 1, 2, 3]
        return manual_set_priorities
        # TODO strategy with overlap area calculation
    
    '''release videoCapure object'''
    def release_all(self):
        cv2.destroyAllWindows()
        n = len(self.frames)
        for idx in range(n):
            print('openable status of {0}th video: input {1}, output {2}'.format(idx, self.frames[idx].isOpened(), self.output_frames[idx].isOpened()))
            self.frames[idx].release()
            self.output_frames[idx].release()
        print('video pos', self.video_pos)
        

TAG_TEST = 2
TAG_NORMAL = 1
DUR_SETUP = "setup the environment"
DUR_TRANS_MASK_CREATION = "create transform area mask"
DUR_INPUT_REDUCE = "create reduced input"
DUR_HUMAN_SEGMENTATION = "create segmentation mask for overlap area"
DUR_OVERLAP = "create the synthesized output image"
DUR_MODEL_PREDICTION = "DNN model prediction(Part of Upper Item - Segmentation mask creation)"
DUR_WHOLE = "overall runtime: "
DUR_ARR = [DUR_SETUP, DUR_TRANS_MASK_CREATION, DUR_INPUT_REDUCE, DUR_HUMAN_SEGMENTATION, DUR_MODEL_PREDICTION]

t_map = {}
t_map[DUR_INPUT_REDUCE] = 0
t_map[DUR_HUMAN_SEGMENTATION] = 0
t_map[DUR_OVERLAP] = 0
t_map[DUR_MODEL_PREDICTION] = 0
def main(offset_frames):
    print("start build up environment and launch")
    start_time = time.time()
    curr_time = start_time
    # calibration input for four views, TODO replace stat with outsource
    calib_stat = masks.stat()
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(curr_dir_path, "../data/")
    video_fps = 20
    ENV = work_env(calib_stat, curr_dir_path, video_path, video_fps)

    tmp_time = time.time()
    t_map[DUR_SETUP] = tmp_time - curr_time
    curr_time = tmp_time

    # pipeline for each frame
    priorities = ENV.get_priorities()
    transforms = ENV.transforms
    homos = ENV.homos
    homo_invs = ENV.homo_invs 
    print("get frame from videos")

    # offset of the original videos
    stats = ENV.get_frames(offset_frames, TAG_NORMAL)
    width = stats[0]
    height = stats[1]
    n = stats[2] # n is the number of views, usually 4 for our dataset
    print("val: {0} {1} {2}".format(width, height, n))

    frames = ENV.frames
    output_frames = ENV.output_frames

    # calculate the static masks
    print("start create overlap mask for frames, shape of transforms={0}, shape of priorities={1}".format(np.shape(transforms), np.shape(priorities)))
    stitch_masks = masks.create_img_mask(height, width, n, transforms, priorities)
    tmp_time = time.time()
    t_map[DUR_TRANS_MASK_CREATION] = tmp_time - curr_time
    curr_time = tmp_time

    ENV.produced_frame_count = 0
    duration = 1
    output_frames = duration * ENV.video_fps # output duration = output frames / 20fps 
    while(ENV.produced_frame_count < output_frames):
        print("start handling frames from {0}/total_frames of the input".format(ENV.video_pos[0]))
        ret = False
        input_images = []
    
        for idx in range(n):
            frames[idx].set(1, ENV.video_pos[idx])
            if not frames[idx].isOpened():
                print("{0}th file is not openable".format(idx))
                exit(1)
            ret, frame = frames[idx].read()
            if frame is None or not ret:
                print("frame lost for {0}th video, exit from the main loop".format(idx))
                break
            input_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # update position of video
        for idx in range(n):
            ENV.video_pos[idx] += ENV.frame_interval
        
        if not ret: 
            print("frame lost for video in {0}th framing".format(ENV.produced_frame_count))
            continue
        
        # use only overlapped area to do human segmentation
        overlapped_input_images = []
        for idx in range(n):
            if idx == priorities[0]:
                overlapped_input_images.append(None)
                continue
            overlapped_input_images.append(cv2.bitwise_and(src1=input_images[idx], src2=(255 - stitch_masks[idx]).astype('uint8')).astype('uint8'))
        
        segmented_masks, tmodel = detect.do_main(ENV, data=overlapped_input_images, priorities=priorities)
        t_map[DUR_MODEL_PREDICTION] += tmodel
        tmp_time = time.time()
        t_map[DUR_HUMAN_SEGMENTATION] += (tmp_time - curr_time)
        curr_time = tmp_time

        reduced_input_images = []

        # TODO 
        # 1, compare video size for same length of video
        # 2, calculate time cost for diff phrases
        for idx in range(n):
            if idx == priorities[0]: 
                reduced_input_images.append(input_images[idx])
                continue
            # reverse mask
            reduced_input_images.append(cv2.bitwise_and(src1=input_images[idx], src2=(stitch_masks[idx]).astype('uint8')))
            # cv2.imwrite('./reduced-' + str(idx) + '.png', reduced_input_images[idx])
        tmp_time = time.time()
        t_map[DUR_INPUT_REDUCE] += (tmp_time - curr_time)
        curr_time = tmp_time
        '''    
        # start video manipulation, apply to each frame 
        output_images = masks.transform_n_crop(reduced_input_images, priorities, stitch_masks, homos, homo_invs)
        output_images = masks.overlap_images(output_images, priorities, segmented_masks, input_images) # overlap the segmented pic onto the transformed image
        '''
        # test use 
        output_images = reduced_input_images
        output_images = masks.overlap_images(output_images, priorities, segmented_masks, input_images) # overlap the segmented pic onto the transformed image
        tmp_time = time.time()
        t_map[DUR_OVERLAP] += (tmp_time - curr_time)
        curr_time = tmp_time

        # np.set_printoptions(threshold=sys.maxsize)
        for idx in range(n):
            # cv2.imwrite('./output-' + str(manual_count) + '-' + str(idx) + '.png', output_images[idx])
            ENV.output_frames[idx].write(output_images[idx]) 
       
        if ret == False: 
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        ENV.produced_frame_count+=1
        
    # remove all 
    ENV.release_all()
    overall_runtime = time.time() - start_time
    print("Tideo has been generated\nvideo offset:\t400\noutput dir:\t{0}\nduration:\t{1}s\nfps:\t{2}\n".format(ENV.data_path, duration, ENV.video_fps))
    print("Time cost for different stage:")
    for event in DUR_ARR:
        print("Action: {0} \t Cost: {1}s \t Percentage: {2}% ".format(event, t_map[event], t_map[event] * 100 / overall_runtime))
    print("------------------------------------------")
    print("Overall Runtime: ", overall_runtime)

def test(offset_frames):
    
    print("program launched with {0} arguments".format(len(sys.argv) - 1))
    # calibration input for four views, TODO replace stat with outsource
    calib_stat = masks.stat()
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(curr_dir_path, "../data/")
    video_fps = 20
    ENV = work_env(calib_stat, curr_dir_path, video_path, video_fps)

    # pipeline for each frame
    priorities = ENV.get_priorities()
    transforms = ENV.transforms
    homos = ENV.homos
    homo_invs = ENV.homo_invs 
    print("get frame from videos")

    # offset of the original videos
    stats = ENV.get_frames(offset_frames, TAG_TEST)
    width = stats[0]
    height = stats[1]
    n = stats[2] # n is the number of views, usually 4 for our dataset
    print("val: {0} {1} {2}".format(width, height, n))

    frames = ENV.frames
    output_frames = ENV.output_frames

    ENV.produced_frame_count = 0
    duration = 1
    output_frames = duration * ENV.video_fps # output duration = output frames / 20fps 
    while(ENV.produced_frame_count < output_frames):
        print("start handling frames from {0}/total_frames of the input".format(ENV.video_pos[0]))
        ret = False
        input_images = []
    
        for idx in range(n):
            frames[idx].set(1, ENV.video_pos[idx])
            if not frames[idx].isOpened():
                print("{0}th file is not openable".format(idx))
                exit(1)
            ret, frame = frames[idx].read()
            if frame is None or not ret:
                print("frame lost for {0}th video, exit from the main loop".format(idx))
                break
            input_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # update position of video
        for idx in range(n):
            ENV.video_pos[idx] += ENV.frame_interval
        
        if not ret: 
            print("frame lost for video in {0}th framing".format(ENV.produced_frame_count))
            continue
        
        for idx in range(n):
            ENV.output_frames[idx].write(input_images[idx]) 
       
        if ret == False: 
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        ENV.produced_frame_count+=1
        
    # remove all 
    ENV.release_all()
    print("video has been generated\nvideo offset:\t400\noutput dir:\t{0}\nduration:\t{1}s\nfps:\t{2}\n".format(ENV.data_path, duration, ENV.video_fps))
    print()
if __name__ == '__main__':
    main(400) 
    # test(400)
