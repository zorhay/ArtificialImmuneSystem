import grey_scale
import scale
from PIL import Image
import numpy as np

def get_bin_array(bimg):
	return np.array(bimg)

if __name__ == '__main__':
	grey_scale.png_to_yuv('../images/auto.jpg', '../images/auto_yuv.jpg')
	im = Image.open("../images/auto_yuv.jpg")
	im = scale.trim(im)
	im.show()