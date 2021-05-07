import code

import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

from skimage import io

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()
    return fig, ax


image = io.imread('Unrailed-2.jpg')
#plt.imshow(image);
#plt.show()

image_slic = seg.slic(image,n_segments=100)
image_show(color.label2rgb(image_slic, image, kind='avg'));




code.interact(local=locals())
