import skimage.io as io
from skimage.color import rgb2gray
from scipy import ndimage as ndi

if __name__ == '__main__':
	im = io.imread("images/im3.jpg")
	im_gray = rgb2gray(im)
	im_diffused = ndi.gaussian_filter(im_gray, 5)
	io.imshow(im_gray, cmap=plt.cm.jet)
	io.show()