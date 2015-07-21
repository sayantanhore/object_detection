from skimage import io, feature
from skimage.color import rgb2gray


if __name__ == '__main__':
	filename = "images/messi.jpeg"
	image = io.imread(filename)
	image_gray = rgb2gray(image)
	io.imshow(image)
	io.show()
	