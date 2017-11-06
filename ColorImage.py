"""Uses:
    python ColorImage.py [Image File Name]"""
import sys
import cv2
import numpy as np

img_name = sys.argv[1]

img = cv2.imread(img_name, cv2.IMREAD_COLOR)

(b, g, r) = cv2.split(img)

cv2.imshow('Original Image', img)
cv2.imshow('Red', r)
cv2.imshow('Green', g)
cv2.imshow('Blue', b)

ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
(y, cb, cr) = cv2.split(ycrcb_image)

cv2.imshow('Y', y)
cv2.imshow('Cr', cr)
cv2.imshow('Cb', cb)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
(h, s, v) = cv2.split(hsv_image)

cv2.imshow('H', h)
cv2.imshow('S', s)
cv2.imshow('V', v)

cv2.imwrite('Original image.png', img)
cv2.imwrite('Red.png', r)
cv2.imwrite('Green.png', g)
cv2.imwrite('Blue.png', b)
cv2.imwrite('Y.png', y)
cv2.imwrite('Cr.png', cr)
cv2.imwrite('Cb.png', cb)
cv2.imwrite('Hue.png', h)
cv2.imwrite('Saturation.png', s)
cv2.imwrite('Value.png', v)

cv2.waitKey(0)
cv2.destroyAllWindows()
