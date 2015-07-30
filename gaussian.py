import sys
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

area_threshold = 500



def draw_border(im, box_dim):
	print("__draw_border__ Enter >>")
	left = box_dim["left"]
	right = box_dim["right"]
	top = box_dim["top"]
	bottom = box_dim["bottom"]
	im[left:right + 1, top] = 1.0
	im[left:right + 1, bottom] = 1.0
	im[left, top:bottom + 1] = 1.0
	im[right, top:bottom + 1] = 1.0
	print("__draw_border__ Exit <<")

def enclose_object(im, box_dim, border_intensity, ended = True):
	print("__enclose_object__ Enter >>")
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
		if (left - 10) >= 0:
			box_dim["left"] = left - 10
		if (right + 10) <= (im.shape[0] - 1):
			box_dim["right"] = right + 10
		if (top - 10) >= 0:
			box_dim["top"] = top - 10
		if (bottom + 10) <= (im.shape[1] - 1):
			box_dim["bottom"] = bottom + 10
	print("__enclose_object__ Exit <<")
	return box_dim


def search_for_object(im, im_gray):
	# Find all the pixels at 60% of maximum intensity of black in the blurred image
	
	max_intensity = np.amax(im)
	print(im.shape)

	while True:
		min_intensity = np.amin(im)
		min_intensity_at = im.argmin()
		object_centre = np.unravel_index(min_intensity_at, im.shape)
		print(str(im.shape[0]) + "::" + str(im.shape[1]))
		print(str(min_intensity) + "::" + str(max_intensity))

		# Check for valid peak
		if min_intensity >= (max_intensity * 40 / 100):
			print("No more peaks")
			break
		'''
		if box_dim not None:
			if ((object_centre[0] >= box_dim["left"]) and (object_centre[0] <= box_dim["right"]) and (object_centre[1] >= box_dim["top"]) and object_centre[1] <= box_dim["bottom"])):
				continue
		'''
		# Get x-y coordinates of the peak
		
		print("Object found at :: " + str(object_centre[0]) + ":" + str(object_centre[1]))


		# Initiate bounding box
		if (object_centre[0] > 0):
			left = object_centre[0] - 1
		else:
			left = object_centre[0]
		if (object_centre[0] < im.shape[0]) - 1:
			right = object_centre[0] + 1
		else:
			right = object_centre[0]
		if (object_centre[1] > 0):
			top = object_centre[1] - 1
		else:
			top = object_centre[1]
		if (object_centre[1] < im.shape[1]) - 1:
			bottom = object_centre[1] + 1
		else:
			bottom = object_centre[1]
		
		
		box_dim = {"top": top, "bottom": bottom, "left": left, "right": right}
		print("Box initiated at :: " + str(box_dim))

		# Create a border intensity threshold
		border_intensity = float(np.amax(im)) * 35 / 100

		# Expand border to enclose object
		box_dim = enclose_object(im, box_dim, border_intensity)
		print("Box updated to :: " + str(box_dim))
		
		# Draw the final border on the image
		box_area = (int(box_dim["right"]) - int(box_dim["left"])) * (int(box_dim["bottom"]) - int(box_dim["top"]))
		print("Box area :: " + str(box_area))
		if box_area > area_threshold:
			draw_border(im_gray, box_dim)

		# Paint detected object in white
		im[box_dim["left"]:box_dim["right"] + 1, box_dim["top"]:box_dim["bottom"] + 1] = 1.0
		print("Object painted")
		
		
	# Returned bordered image
	return im_gray

if __name__ == '__main__':
	im_index = sys.argv[1]
	im = io.imread(path + "images/im" + im_index + ".jpg")
	im_gray = rgb2gray(im)
	im_gauss = gf(im_gray, sigma = 10)
	im_edge = sobel(im_gauss)
	blobs_doh = blob_doh(im_gray, max_sigma=30, threshold=.01)
	print("-------------------")
	#print(im_edge.shape)
	#print(np.amax(im_gauss))
	#im_edge.fill(0)
	# Search for object
	im_modified = search_for_object(im_gauss.copy(), im_gray.copy())
	print("-------------------")
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = rows, ncols = cols, figsize = (fig_x, fig_y))
	ax1.imshow(im)
	ax2.imshow(im_gauss, cmap = plt.cm.gray)
	ax3.imshow(im_edge, cmap = plt.cm.gray)
	ax4.imshow(im_modified, cmap = plt.cm.gray)
	plt.show()

