"""Uses:
    python Threshold.py [Image File Name]"""

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_name = sys.argv[1]
img = cv2.imread(img_name, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('Input Image', img)

threshold_type = 2
threshold_value = 128

ret, dst = cv2.threshold(gray, threshold_value, 255, threshold_type)

cv2.imshow('Thresholded image', dst)

current_threshold = 128
max_threshold = 255


ret, thresholded = cv2.threshold(
    gray, current_threshold, max_threshold, cv2.THRESH_BINARY)

cv2.imshow('Binary threshold', thresholded)

threshold1 = 27
threshold2 = 125

ret, t1 = cv2.threshold(gray, threshold1, max_threshold, cv2.THRESH_BINARY)
ret, t2 = cv2.threshold(gray, threshold2, max_threshold, cv2.THRESH_BINARY_INV)

t3 = t1 & t2

cv2.imshow('Band Thresholding', t3)

ret, semi_thresholded_image = cv2.threshold(
    gray, current_threshold, max_threshold, cv2.THRESH_BINARY_INV)
semi_thresholded_image = semi_thresholded_image & gray
cv2.imshow('Semi Thresholding', semi_thresholded_image)

adaptive_thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    101,
    10)
cv2.imshow('Adaptive Thresholding', adaptive_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
