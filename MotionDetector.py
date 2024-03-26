# Hien Dao
# CSE 4310 - Computer Vision

import numpy as np
import cv2
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from KalmanFilter import KalmanFilter

'''
    Motion Detector Class:
    a - Frame hysteresis for determining active or inactive objects.
    t - The motion threshold for filtering out noise.
    d - A distance threshold to determine if an object candidate belongs to an object currently being tracked.
    s - The number of frames to skip between detections. The tracker will still work well even if it is not updated every frame.
    N - The maximum number of objects to track.
    KF - Kalman Filter used to track object.
'''

class MotionDetector():
    def __init__(self, a, t, d, s, N, KF):
        self.a = a
        self.t = t
        self.d = d
        self.s = s
        self.N = N
        self.KF = KF
        self.objects = []
        self.active_objects = []

    def update(self, frames, idx):
        # Skip initialization frames
        if idx <= 3:
            return
        
        # Convert frames from rgb to grayscale
        ppframe = rgb2gray(frames[idx-2])
        pframe = rgb2gray(frames[idx-1])
        cframe = rgb2gray(frames[idx])

        # Compute difference between the frames
        diff1 = np.abs(cframe - pframe)
        diff2 = np.abs(pframe - ppframe)

        motion_frame = np.minimum(diff1, diff2)
        thresh_frame = motion_frame > self.t
        dilated_frame = dilation(thresh_frame, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        
        # Detect motion for object candidates and adds to list of candidates
        candidates = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox

            (x1,y1) = self.KF.predict()
            (x2,y2) = self.KF.update([[np.mean([minr,maxr])], [np.mean([minc,maxc])],[0],[0]])

            if abs(x2-x1) <= self.d and abs(y2-y1) <= self.d:
                cv2.rectangle(frames[idx], (minc, minr), (maxc, maxr), (0, 0, 255), 2)
                candidates.append(self.KF)

        for obj in self.active_objects:
            obj.frames_inactive += 1

        for i, candidate in enumerate(candidates):
            matched = False
            for j, obj in enumerate(self.objects):
                x2 = candidate.prev_positions[i][0]
                y2 = candidate.prev_positions[i][1]
                x1 = obj.prev_positions[j][0]
                y1 = obj.prev_positions[j][1]

                # Check if candidate matches an existing object based on distance threshold
                if abs(x2-x1) <= self.d and abs(y2-y1) <= self.d:
                    obj.frames_active += 1
                    obj.frames_inactive = 0
                    matched = True
                    break

            # If candidate has no match then add to list of objects proposals
            if not matched:
                self.objects.append(self.KF)

        # If an object proposal is active over 'a' frames, it should be added to the list of currently tracked objects.
        for obj in self.objects:
            if obj.frames_active >= self.a and len(self.active_objects) < self.N:
                self.active_objects.append(obj)

        # If a currently tracked object has not been updated with a measurement in 'a' frames, consider it to be inactive and remove the filter from the current list of tracked objects
        for obj in self.active_objects:
            if obj.frames_inactive >= self.a:
                self.active_objects.remove(obj)
