import cv2
import math
import numpy as np


class VectorField:
    def __init__(self, fieldx, fieldy):
        self.fieldx = fieldx
        self.fieldy = fieldy

    @staticmethod
    def from_gradient(gray):
        fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
        fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

        return VectorField(fieldx, fieldy)

    def get_magnitude_image(self):
        res = np.sqrt(self.fieldx**2 + self.fieldy**2)
        
        return (res * 255/np.max(res)).astype(np.uint8)

    def smooth(self, radius, iterations=1):
        s = 2*radius + 1
        for _ in range(iterations):
            self.fieldx = cv2.GaussianBlur(self.fieldx, (s, s), 0)
            self.fieldy = cv2.GaussianBlur(self.fieldy, (s, s), 0)

    def direction(self, i, j):
        return math.atan2(self.fieldy[i, j], self.fieldx[i, j])

    def magnitude(self, i, j):
        return math.hypot(self.fieldx[i, j], self.fieldy[i, j])
