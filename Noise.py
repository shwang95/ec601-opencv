"""Uses:
    python Noise.py [Image File Name]"""

import sys
import cv2
from numpy import *
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 5})


def add_gaussian_noise(src, mean, sigma):
    noiseArr = src.copy()
    noiseArr = random.normal(mean, sigma, src.shape)
    add(src, noiseArr, src, casting="unsafe")
    return


def add_salt_pepper_noise(src, pa, pb):
    amount1 = (int)(src.size * pa)
    amount2 = (int)(src.size * pb)
    for counter in range(0, amount1):
        src[(int)(random.uniform(0, src.shape[0] - 1))][(int)
                                                        (random.uniform(0, src.shape[1] - 1))] = 0
    for counter in range(0, amount2):
        src[(int)(random.uniform(0, src.shape[0] - 1))][(int)
                                                        (random.uniform(0, src.shape[1] - 1))] = 255


img_name = sys.argv[1]
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', img)

kernel = [(3, 3), (5, 5), (7, 7)]
mean = [0, 5, 10, 20]
sigma = [0, 20, 50, 100]
index = 0
for k in range(0, 3):
    for i in range(0, 4):
        for j in range(0, 4):
            plt.figure(index)
            plt.subplot(2, 2, 1)
            noise_img = img.copy()
            name = 'Gaussian Noise,\n mean: %d, sigma: %d' % (
                mean[i], sigma[j])
            add_gaussian_noise(noise_img, mean[i], sigma[j])
            plt.imshow(noise_img, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 2)
            noise_dst = noise_img.copy()
            name = 'Box filter,\n mean: %d, sigma: %d, kernel: %dx%d' % (
                mean[i], sigma[j], kernel[k][0], kernel[k][0])
            cv2.blur(noise_dst, kernel[k], noise_dst)
            plt.imshow(noise_dst, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 3)
            noise_dst1 = noise_img.copy()
            name = 'Gaussian filter,\n mean: %d, sigma: %d, kernel: %dx%d' % (
                mean[i], sigma[j], kernel[k][0], kernel[k][0])
            cv2.GaussianBlur(noise_img, kernel[k], 1.5, noise_dst1)
            plt.imshow(noise_dst1, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 4)
            noise_dst2 = noise_img.copy()
            name = 'Median filter,\n mean: %d, sigma: %d, kernel: %dx%d' % (
                mean[i], sigma[j], kernel[k][0], kernel[k][0])
            cv2.medianBlur(noise_img, kernel[k][0], noise_dst2)
            plt.imshow(noise_dst2, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])
            plt.draw()
            plt.savefig(
                'Gaussian_mean%d_sigma%d_kernel%dx%d.png' %
                (mean[i], sigma[j], kernel[k][0], kernel[k][0]))
            index = index + 1


pa = [0.01, 0.03, 0.05, 0.4]
pb = [0.01, 0.03, 0.05, 0.4]
for k in range(0, 3):
    for i in range(0, 4):
        for j in range(0, 4):
            plt.figure(index)
            plt.subplot(2, 2, 1)
            noise_img2 = img.copy()
            name = 'Salt and Pepper Noise,\n pa: %f, pb: %f' % (pa[i], pb[j])
            add_salt_pepper_noise(noise_img2, pa[i], pb[j])
            plt.imshow(noise_img2, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 2)
            noise_dst3 = noise_img2.copy()
            name = 'Box filter,\n pa: %f, pb: %f, kernel: %dx%d' % (
                pa[i], pb[j], kernel[k][0], kernel[k][0])
            cv2.blur(noise_dst3, kernel[k], noise_dst3)
            plt.imshow(noise_dst3, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 3)
            noise_dst4 = noise_img2.copy()
            name = 'Gaussian filter,\n pa: %f, pb: %f, kernel: %dx%d' % (
                pa[i], pb[j], kernel[k][0], kernel[k][0])
            cv2.GaussianBlur(noise_dst4, kernel[k], 1.5, noise_dst4)
            plt.imshow(noise_dst4, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 4)
            noise_dst5 = noise_img2.copy()
            name = 'Median filter,\n pa: %f, pb: %f, kernel: %dx%d' % (
                pa[i], pb[j], kernel[k][0], kernel[k][0])
            cv2.medianBlur(noise_dst5, kernel[k][0], noise_dst5)
            plt.imshow(noise_dst5, 'gray')
            plt.title(name)
            plt.xticks([]), plt.yticks([])
            plt.draw()
            plt.savefig(
                'SaltAndPepper_pa%f_pb%f_kernel%dx%d.png' %
                (pa[i], pb[j], kernel[k][0], kernel[k][0]))
            index = index + 1

plt.show()
