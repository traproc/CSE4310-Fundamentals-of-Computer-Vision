# Hien Dao 1001912046
# CSE 4310 - Fundamentals of Computer Vision

import numpy as np
from numpy.lib import stride_tricks
import random
import skimage.io as io
import matplotlib.pyplot as plt

def random_crop(img, size):
    height, width = img.shape[0], img.shape[1]
    if (size > 0) and (size <= min(height, width)):
        # Generate random (x,y) location
        x1 = random.randint(0, width-size)
        y1 = random.randint(0, height-size)

        # Compute (x,y) with respect to size input and crop
        x2 = x1 + size
        y2 = y1 + size
        img_crop = img[x1:x2, y1:y2]
        return img_crop
    else:
        print("ERROR: Crop size is larger than input image\n")
        exit(0)

def extract_patch(img, num_patches):
    # Non-overlapping patches of size -> num_patches
    H, W, channels = img.shape[0], img.shape[1], img.shape[2]
    patch_h, patch_w = H // num_patches, W // num_patches
    shape = [num_patches,num_patches] + [patch_h, patch_w, channels]

    # (row, col, patch_row, patch_col)
    strides = [patch_h * s for s in img.strides[:2]] + list(img.strides)

    # Extract patches
    patches = stride_tricks.as_strided(img, 
                                       shape=shape, 
                                       strides=strides)

    # Save each patch image
    for i in range(num_patches):
        for j in range(num_patches):
            patch_img = patches[i,j]
            io.imsave("patch_"+str(i)+"_"+str(j)+".jpg", patch_img.astype(np.uint8))

def resize_img(img, factor):
    # Get old dimensions from image
    height, width = img.shape[0], img.shape[1]

    # Get new dimensions from scaling original image
    new_height = int(height * factor)
    new_width = int(width * factor)

    # Initialize resize image array
    resize_img = np.zeros((new_height, new_width, img.shape[2]),dtype=img.dtype)

    for i in range(new_height):
        for j in range(new_width):
            resize_img[i][j] = img[int(i/factor)][int(j/factor)]

    return resize_img

def rgb_to_hsv(rgb_hsv):
  # Normalize RGB values to be in range of [0,1]
  rgb_hsv = rgb_hsv/255
  r, g, b = rgb_hsv[:,:,0], rgb_hsv[:,:,1], rgb_hsv[:,:,2]

  # Calculating HSV value V
  v = np.max(rgb_hsv, axis=2)

  # Calculating Saturation with Chroma
  chroma = v - np.min(rgb_hsv, axis=2)
  s = np.where(v!=0, chroma/v, 0)

  # Calculating Hue
  # if chroma == 0 in else condition below
  # if v == r:
  hue_comp = np.where((chroma != 0) & (v==r),((g-b)/chroma)%6, 0)
  # if v == g:
  hue_comp = np.where((chroma != 0) & (v==g),((b-r)/chroma)+2, hue_comp)
  # if v == b:
  hue_comp = np.where((chroma != 0) & (v==b),((r-g)/chroma)+4, hue_comp)

  h = 60.0 * hue_comp
  rgb_hsv = np.stack((h,s,v), axis = -1)
  return rgb_hsv
  
    
def hsv_to_rgb(hsv_rgb):
  h,s,v = hsv_rgb[:,:,0], hsv_rgb[:,:,1], hsv_rgb[:,:,2]
  h = np.where(h >= 360, h-360, h)
  chroma = v * s
  hue_comp = h / 60
  x = chroma * (1 - abs((hue_comp % 2) - 1))

  # Calculate (R',G',B')
  R_comp =  np.where((hue_comp >= 0) & (hue_comp < 1), chroma,
            np.where((hue_comp >= 1) & (hue_comp < 2), x,
            np.where((hue_comp >= 2) & (hue_comp < 3), 0,
            np.where((hue_comp >= 3) & (hue_comp < 4), 0,
            np.where((hue_comp >= 4) & (hue_comp < 5), x,
            np.where((hue_comp >= 5) & (hue_comp < 6), chroma,0
            ))))))
  
  G_comp =  np.where((hue_comp >= 0) & (hue_comp < 1), x,
            np.where((hue_comp >= 1) & (hue_comp < 2), chroma,
            np.where((hue_comp >= 2) & (hue_comp < 3), chroma,
            np.where((hue_comp >= 3) & (hue_comp < 4), x,
            np.where((hue_comp >= 4) & (hue_comp < 5), 0,
            np.where((hue_comp >= 5) & (hue_comp < 6), 0,0
            ))))))
  
  B_comp =  np.where((hue_comp >= 0) & (hue_comp < 1), 0,
            np.where((hue_comp >= 1) & (hue_comp < 2), 0,
            np.where((hue_comp >= 2) & (hue_comp < 3), x,
            np.where((hue_comp >= 3) & (hue_comp < 4), chroma,
            np.where((hue_comp >= 4) & (hue_comp < 5), chroma,
            np.where((hue_comp >= 5) & (hue_comp < 6), x,0
            ))))))
  
  m = v - chroma

  # Calculate (R,G,B)
  r = (R_comp + m) * 255
  g = (G_comp + m) * 255
  b = (B_comp + m) * 255
  hsv_rgb = np.stack((r,g,b), axis = -1)
  return hsv_rgb

def color_jitter(img, hue, saturation, value):
    if hue < 0 or hue > 360:
        print("The Hue input is out of the range [0,360]")
        exit(0)
    elif saturation < 0 or saturation > 1:
        print("The Saturation input is out of the range [0,1]")
        exit(0)
    elif value < 0 or value > 1:
        print("The Value input is out of the range [0,1]")
        exit(0)

    # Calculate random HSV values by an amount no greater than the given input
    hue = random.uniform(0,hue)
    saturation = random.uniform(0,saturation)
    value = random.uniform(0,value)

    # RGB to HSV
    hsv_img = rgb_to_hsv(img)

    # Command Line HSV Modification Values
    hsv_img[:,:,0] += hue
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] + saturation, 0, 1)
    hsv_img[:,:,2] = np.clip(hsv_img[:,:,2] + value, 0, 1)

    # HSV to RGB
    rgb_img = hsv_to_rgb(hsv_img)
    return rgb_img

img = io.imread("image.jpg")

crop_image = random_crop(img,20)
io.imshow(crop_image)
plt.show()

resized_image = resize_img(img,2)
io.imshow(resized_image)
plt.show()

jitter_image = color_jitter(img,100,0.5,0)
io.imshow(jitter_image.astype(np.uint8))
plt.show()

extract_patch(img,3)