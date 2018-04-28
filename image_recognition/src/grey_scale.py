from PIL import Image

def png_to_yuv(input_image, output_image):
	img = Image.open(input_image)
	img_yuv = img.convert('L')
	point = calc_eta(img_yuv)
	bw = img_yuv.point(lambda x: 0 if x<point else 255, '1')
	bw.save(output_image)

def calc_eta(img):
	width,height = img.size
	total = 0
	for i in range(0,width):
		for j in range(0,height):
			total += img.getpixel((i,j))
	mean = total / (width * height)
	return mean

if __name__ == '__main__':
	# png_to_yuv('../images/cat.jpg', '../images/cat_yuv.jpg')
	# png_to_yuv('../images/rick.jpg', '../images/rick_yuv.jpg')
	# png_to_yuv('../images/rub.jpg', '../images/rub_yuv.jpg')
	png_to_yuv('../images/dav.jpg', '../images/dav_yuv.jpg')

