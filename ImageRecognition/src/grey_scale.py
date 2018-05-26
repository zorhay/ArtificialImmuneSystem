from PIL import Image
import numpy

def png_to_yuv(input_image, output_image=None):
    img = Image.open(input_image)
    img_yuv = img.convert('L')
    point = calc_eta(img_yuv)
    bw = img_yuv.point(lambda x: 0 if x<point else 255, '1')
    if output_image:
        bw.save(output_image)
    return bw

def calc_eta(img):
    width,height = img.size
    total = 0
    for i in range(0,width):
        for j in range(0,height):
            total += img.getpixel((i,j))
    mean = total / (width * height)
    return mean


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


if __name__ == '__main__':
    # png_to_yuv('../images/cat.jpg', '../images/cat_yuv.jpg')
    # png_to_yuv('../images/rick.jpg', '../images/rick_yuv.jpg')
    # png_to_yuv('../images/rub.jpg', '../images/rub_yuv.jpg')
    png_to_yuv('../images/dav.jpg', '../images/dav_yuv.jpg')


































