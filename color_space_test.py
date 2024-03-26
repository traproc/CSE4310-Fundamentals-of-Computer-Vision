# Hien Dao 1001912046
# CSE 4310 - Fundamentals of Computer Vision

import sys
import numpy as np
import skimage.io as io

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

filename = sys.argv[1]
hue = float(sys.argv[2])
saturation = float(sys.argv[3])
value = float(sys.argv[4])

if hue < 0 or hue > 360:
  print("The Hue input is out of the range [0,360]")
  exit(0)
elif saturation < 0 or saturation > 1:
  print("The Saturation input is out of the range [0,1]")
  exit(0)
elif value < 0 or value > 1:
  print("The Value input is out of the range [0,1]")
  exit(0)

img = io.imread(filename)

# RGB to HSV
hsv_img = rgb_to_hsv(img)

# Command Line HSV Modification Values
hsv_img[:,:,0] = hsv_img[:,:,0] + hue
hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] + saturation, 0, 1)
hsv_img[:,:,2] = np.clip(hsv_img[:,:,2] + value, 0, 1)

# HSV to RGB
rgb_img = hsv_to_rgb(hsv_img)

io.imsave("new_image.jpg", rgb_img.astype(np.uint8))