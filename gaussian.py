from path import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import blob_doh
from skimage import io
from skimage.filters import sobel, gaussian_filter as gf

rows = 1
cols = 4
fig_x = 8
fig_y = 4

def draw_border(im, box_dim):
	left = box_dim["left"]
	right = box_dim["right"]
	top = box_dim["top"]
	bottom = box_dim["bottom"]
	im[left:right, top] = 1.0
	im[left:right, bottom] = 1.0
	im[left, top:bottom] = 1.0
	im[right, top:bottom] = 1.0

def enclose_object(im, box_dim, border_intensity, ended = True):
	left = box_dim["left"]
	right = box_dim["right"]
	top = box_dim["top"]
	bottom = box_dim["bottom"]

	for key in box_dim.keys():
		if key == "left":
			x = box_dim[key] - 10
			if x >= 0:
				slice = im[x:left, top:bottom]
				# Check for pixel
				found = slice[slice < border_intensity].size
				if found > 0:
					box_dim[key] = x
					left = x
					ended = False
		if key == "right":
			x = box_dim[key] + 10
			if x <= im.shape[0] - 1:
				slice = im[right:x, top:bottom]
				# Check for pixel
				found = slice[slice < border_intensity].size
				if found > 0:
					box_dim[key] = x
					right = x
					ended = False
		if key == "top":
			y = box_dim[key] - 10
			if y >= 0:
				slice = im[left:right, y:top]
				# Check for pixel
				found = slice[slice < border_intensity].size
				if found > 0:
					box_dim[key] = y
					top = y
					ended = False
		if key == "bottom":
			y = box_dim[key] + 10
			if y <= im.shape[1] - 1:
				slice = im[left:right, bottom:y]
				# Check for pixel
				found = slice[slice < border_intensity].size
				if found > 0:
					box_dim[key] = y
					bottom = y
					ended = False

	if not ended:
		enclose_object(im, box_dim, border_intensity)
	else:
		box_dim["left"] = left - 10
		box_dim["right"] = right + 10
		box_dim["top"] = top - 10
		box_dim["bottom"] = bottom + 10
	
	return box_dim


def search_for_object(filter_dim, im, im_gray):
	# Find all the pixels at 60% of maximum intensity of black in the blurred image
	max_intensity = im.argmin()
	print(im.shape)
	object_centre = np.unravel_index(max_intensity, im.shape)
	print("Object found at :: " + str(object_centre[0]) + ":" + str(object_centre[1]))
	# Initiate bounding box
	top = object_centre[1] - 1
	bottom = object_centre[1] + 1
	left = object_centre[0] - 1
	right = object_centre[0] + 1
	box_dim = {"top": top, "bottom": bottom, "left": left, "right": right}
	border_intensity = float(np.amax(im_gauss)) * 40 / 100
	box_dim = enclose_object(im, box_dim, border_intensity)
	print(box_dim)
	print(im.shape[0])
	draw_border(im_gray, box_dim)
	return im_gray

if __name__ == '__main__':
	im = io.imread(path + "images/im4.jpg")
	im_gray = rgb2gray(im)
	im_gauss = gf(im_gray, sigma = 10)
	im_edge = sobel(im_gauss)
	blobs_doh = blob_doh(im_gray, max_sigma=30, threshold=.01)
	print("-------------------")
	#print(im_edge.shape)
	#print(np.amax(im_gauss))
	#im_edge.fill(0)
	# Search for object
	im_modified = search_for_object([100, 100], im_gauss, im_gray.copy())
	print("-------------------")
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = rows, ncols = cols, figsize = (fig_x, fig_y))
	ax1.imshow(im)
	ax2.imshow(im_gray, cmap = plt.cm.gray)
	ax3.imshow(im_gauss, cmap = plt.cm.gray)
	ax4.imshow(im_modified, cmap = plt.cm.gray)
	plt.show()