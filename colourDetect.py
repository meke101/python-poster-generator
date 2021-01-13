# https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6
if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt

    # %matplotlib inline

    pic = imageio.imread('squares-color.png')
    plt.figure(figsize=(5, 5))
    plt.imshow(pic)

    # observate props
    print('Type of the image : ', type(pic))
    print()
    print("Shape of the image : {}".format(pic.shape))
    print('Image Hight {}'.format(pic.shape[0]))
    print('Image Width {}'.format(pic.shape[1]))
    print('Dimension of Image {}'.format(pic.ndim))
    print('Image size {}'.format(pic.size))
    print('Maximum RGB value in this image {}'.format(pic.max()))
    print('Minimum RGB value in this image {}'.format(pic.min()))

    # specific pixel colour at rows, cols
    print(pic[400, 300])
    # Image([109, 143, 46], dtype=uint8)

from PIL import Image

basewidth = 420
img = Image.open('squares-color.png')
print("W, H, before scale", img.size)
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img.save('scaled_image.png')
print("W, H, after scale", img.size)

# https://opensource.com/life/15/2/resize-images-python
# # https://code-maven.com/crop-images-using-python-pil-pillow
from PIL import ImageFont
from PIL import ImageDraw
import sys
import numpy as np
import pandas as pd
import cv2

in_file = 'scaled_image.png'
out_file = 'cropped_and_scaled.png'

img = Image.open(in_file)
width, height = img.size

# crop
# 10 pixels from the left
# 20 pixels from the top
# 30 pixels from the right
# 40 pixels from the bottom
print(">>", img.size)
height_crop = (594-height) / 2
print("crop off both sides for A2", height_crop)
cropped = img.crop((0, -height_crop, width - 0, height + height_crop))
cropped.save(out_file)
croppedImg = Image.open(out_file)
print("cropped img size", croppedImg.size)


#
# from __future__ import print_function
# import binascii
# import struct
# from PIL import Image
# import numpy as np
# import scipy
# import scipy.misc
# import scipy.cluster
#
# NUM_CLUSTERS = 5
#
# print('reading image')
# im = Image.open('cropped.png')
# im = im.resize((150, 150))      # optional, to reduce time
# ar = np.asarray(im)
# shape = ar.shape
# ar = ar.reshape(np.product(shape[1]), shape[1]).astype(float)
#
# print('finding clusters')
# codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
# print('cluster centres:\n', codes)
#
# vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
# counts, bins = np.histogram(vecs, len(codes))    # count occurrences
#
# index_max = np.argmax(counts)                    # find most frequent
# peak = codes[index_max]
# colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
# print('most frequent is %s (#%s)' % (peak, colour))
# print("HERE" , colour)
