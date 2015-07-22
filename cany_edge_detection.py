import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature, io, exposure
from skimage.color import rgb2gray

from path import *


im_orig = io.imread(path + "images/im3.jpg")
im_gray = rgb2gray(im_orig)
#im = exposure.equalize_hist(im)

#im = ndi.rotate(im, 15, mode='constant')
im = ndi.gaussian_filter(im_gray, 5)
im += 0.2 * np.random.random(im.shape)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(im_orig)
ax1.axis('off')
ax1.set_title('Original image', fontsize=20)

ax2.imshow(im_gray, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Grayscale image', fontsize=20)

'''
ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=3$', fontsize=20)
'''

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()