# Hien Dao
# CSE 4310 - Computer Vision

import numpy as np

class KalmanFilter():

    def __init__(self, t):

        self.t = t
        
        # u - control vector
        self.u = np.zeros((2, 1))

        self.x = np.zeros((4, 1))

        self.D = np.matrix([[1, 0, self.t, 0],
                            [0, 1, 0, self.t],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
                            
        # E - Covariance Matrix
        self.E = np.eye(len(self.D))

        # B - control matrix
        self.B = np.matrix([[(self.t**2)/2, 0],
                            [0,(self.t**2)/2],
                            [self.t,0],
                            [0,self.t]])

        # N - Noise Covariance
        self.N = np.diag([0.1, 0.1, 0.1, 0.1])

        self.frames_inactive = 0
        self.frames_active = 0
        self.box = None
        self.prev_positions = []

    def predict(self):
        # x is the prediction of our current state based on the previous best estimate with an added correction term based on known factors (acceleration). 
        self.x = np.dot(self.D, self.x) + np.dot(self.B, self.u)

        # E is the updated uncertainty based on the old uncertainty with added Gaussian noise to reflect unknown factors.
        self.E = np.dot(np.dot(self.D, self.E), self.D.T) + self.N

        return int(self.x[0]), int(self.x[1])

    def update(self, y):
        # Calculate Kalman Gain
        K = np.dot(np.dot(self.D, np.dot(self.E, self.D.T)), np.linalg.inv(np.dot(self.D, np.dot(self.E, self.D.T)) + self.N))
        self.x = np.dot(self.D, self.x) + np.dot(K, (y - np.dot(self.D, self.x)))
        self.E = np.dot(self.D, np.dot(self.E, self.D.T)) - np.dot(K, np.dot(self.D, np.dot(self.E, self.D.T)))
        self.prev_positions.append((int(self.x[0]), int(self.x[1])))
        return int(self.x[0]), int(self.x[1])
    
