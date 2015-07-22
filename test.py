import skimage.io as io
from skimage.color import rgb2gray
from scipy import ndimage as ndi


from path import path

if __name__ == '__main__':
	im = io.imread(path + "images/im3.jpg")
	im_gray = rgb2gray(im)
	im_diffused = ndi.gaussian_filter(im_gray, 5)
	io.imshow(im_gray)
	io.show()