# Hien Dao 1001912046
# CSE 4310 - Fundamentals of Computer Vision

import numpy as np
import skimage.io as io

def resize_img(img, factor):
    # Get old dimensions from image
    height, width = img.shape[0], img.shape[1]

    # Get new dimensions from scaling original image
    new_height = int(height * factor)
    new_width = int(width * factor)

    resize_img = np.zeros((new_height, new_width, img.shape[2]),dtype=img.dtype)

    for i in range(new_height):
        for j in range(new_width):
            resize_img[i][j] = img[int(i/factor)][int(j/factor)]

    return resize_img

def image_pyramid(img, p_height):
    for i in range(1,p_height):
        img = resize_img(img,0.5)
        io.imsave("image_"+str(2**i)+"x.jpg",img)

img = io.imread("image.jpg")
image_pyramid(img,4)