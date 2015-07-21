import matplotlib.pyplot as plt
import numpy as np


from skimage import io, feature, exposure
from skimage.feature import corner_harris, corner_peaks, blob_dog
from skimage.color import rgb2gray

def show_marked_image(marks, image, title = None):
	fig = plt.figure()
	plt.imshow(image)
	y_corners, x_corners, z_corners = zip(*marks)
	plt.plot(x_corners, y_corners, z_corners, "-")
	plt.show()

if __name__ == '__main__':
	filename = "images/messi.jpeg"
	image = io.imread(filename)
	image_gray = rgb2gray(image)
	
	
	#marks = corner_peaks(corner_harris(image_gray), min_distance = 2)
	
	test_image = exposure.equalize_hist(image_gray)

	marks = blob_dog(test_image, threshold = 0.7)
	print(marks)
	
	show_marked_image(marks, image)
	

	'''
	io.imshow(image_gray)
	io.show()
	'''

