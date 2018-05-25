"""
This file contains a implement of Reading data from a video and then converting them into some groups.

!!! Need to know in line 54. I have no time to review the code to make following step innecessary. :(
!!! When you want to test maptv.py, line 54 should be 'frames_cur = cv.cvtColor(self.v.read()[1], cv.COLOR_BGR2RGB)'
!!! When you want to test srcnn.py, line 54 should be 'frames_cur = cv.cvtColor(self.v.read()[1], cv.COLOR_BGR2YCrCb)'
"""

import cv2 as cv
import numpy as np


class FramesBatch:
    # A object of batch extracted from video file

    def __init__(self,
                 data: 'Inputdata of batch. Its shape should be a format like [height, width, frames]',
                 key_index: 'Position of Keyframe in batch(middle[defalut])'):
        self.data = data
        self.key_index = key_index

    def set_keyframe(self, data):  # Write keyframe
        self.data[:, :, self.key_index] = data

    def get_keyframe(self):  # Read keyframe
        return self.data[:, :, self.key_index]


class VideoSRContinuous:

    def __init__(self,
                 path: 'Path of video file',
                 channel,
                 batchframe_size: 'size of each frame in batch',
                 batchframe_num: 'frames in each batch ' = 9,
                 ):

        self.batchframe_num = batchframe_num
        self.batchframe_size = batchframe_size

        self.v = cv.VideoCapture(path)  # video object obtained by opencv api
        self.height = int(self.v.get(cv.CAP_PROP_FRAME_HEIGHT))  # original height of frame in video
        self.width = int(self.v.get(cv.CAP_PROP_FRAME_WIDTH))  # original width of frame in video
        self.totalframes_num = int(self.v.get(cv.CAP_PROP_FRAME_COUNT))  # how many frame stores in original video

        frames_raw = np.zeros(shape=[self.height, self.width, self.totalframes_num])
        # store raw data of video

        frames_crop = np.zeros(shape=[self.batchframe_size[0], self.batchframe_size[1], self.totalframes_num])
        # store croped data of video

        # make shape of frame in video square
        for i in range(self.totalframes_num):
            frames_cur = cv.cvtColor(self.v.read()[1], cv.COLOR_BGR2RGB)
            frames_raw[:, :, i] = frames_cur[:, :, channel]

        if self.height == self.width:
            for i in range(self.totalframes_num):
                frames_crop[:, :, i] = cv.resize(frames_raw[:, :, i], self.batchframe_size)
        else:
            if self.height > self.width:
                for i in range(self.totalframes_num):
                    frames_crop[:, :, i] = cv.resize(frames_raw[:self.width, :, i], self.batchframe_size)
            else:
                for i in range(self.totalframes_num):
                    frames_crop[:, :, i] = cv.resize(frames_raw[:, :self.height, i], self.batchframe_size)

        self.batches = []
        # store batches data; croped data of video will be divied into many batches; in each batch, there is
        # batchframe_num frames that are near by each other in video file.

        for index in range(self.totalframes_num - self.batchframe_num):
            index_next = index + self.batchframe_num
            self.batches.append(FramesBatch(data=frames_crop[:, :, index:index_next],
                                            key_index=int(self.batchframe_num/2)))

    def totalframes(self):
        # Get data of total frames from 'VideoSRContinuous' object
        out = np.zeros(shape=[self.batchframe_size[0], self.batchframe_size[1], self.batches.__len__()])

        index = 0
        for i in range(self.batches.__len__()):
            index_next = index + self.batchframe_num
            out[:, :, i] = self.batches[i].get_keyframe().copy()
            index = index_next

        return out