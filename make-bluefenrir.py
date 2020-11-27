import numpy as np
from skimage import io, color, img_as_float

image = io.imread('fenrir.png')
# print(image.shape)
# image is 4-channel with 400x600 size

# convert image to 3-channel array at LAB space
image_rgb = color.rgba2rgb(image)
image_lab = color.rgb2lab(image_rgb) 

# LAB color values
lab_colors = {
    'vermillion': [44.441, 52.797, 43.768],
    'red_orange': [45.824, 44.129, 47.554],
    'tomato_red': [35.45, 43.402, 30.523],
    'deep_orange': [59.241, 40.856, 64.504],
    'raspberry_red': [38.686, 53.68, 20.868],
    'wine_red': [19.699, 30.019, 12.525],
    'purple_red': [23.903, 35.433, 16.085],
    'claret_violet': [23.577, 34.298, 0.517],
}

# create mask of pixels that are close to any color in lab_colors
mask = np.full(image.shape[:2], False)
for lab_color in lab_colors:
    mask |=  color.deltaE_ciede2000(image_lab, lab_colors[lab_color])<15

# mask out the eyes
mask[295:303, 153:163].fill(False)
mask[293:300, 183:193].fill(False)

# convert the image to HSV space and change the hue(H) of the pixels in mask
image_hsv = color.rgb2hsv(image_rgb)
image_hsv[mask] += np.array([0.6,0,0])

# save the created image
image_rgb = color.hsv2rgb(image_hsv)

io.imsave(
    'fenrir-blue.png',
    np.asarray(image_rgb, dtype=np.float),
)
