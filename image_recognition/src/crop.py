from PIL import Image
from scale import trim
import os
import numpy
from grey_scale import binarize_array, binarize_image

image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSetCopy/'
training_set = {
                '0': image_path + '0/',
                '1': image_path + '1/',
                '2': image_path + '2/',
                '3': image_path + '3/',
                '4': image_path + '4/',
                '5': image_path + '5/',
                '6': image_path + '6/',
                '7': image_path + '7/',
                '8': image_path + '8/',
                '9': image_path + '9/'
                }

def MyFilter(Box):
    temp_arr = [0 for i in range(9)]
    counter = 0
    for v in range(3):
        for h in range(3):
            temp_arr[counter] = Box[v][h]
            counter = counter + 1
    temp_arr = numpy.sort(temp_arr)
    return temp_arr[4]

def filter(path):
    img = Image.fromarray(binarize_image(path, None, 160), 'L')
    img1 = numpy.array(img, dtype=numpy.uint8)
    Matrix = [[0 for x in range(img.size[0])] for y in range(img.size[1])]

    for i in range(img.size[0] - 2):
        for y in range(img.size[1] - 2):
            local = img1[i:i + 3, y:y + 3]
            Matrix[i + 1][y + 1] = MyFilter(local)

    # finish part
    image_matrix = numpy.uint8(Matrix)
    image = Image.fromarray(image_matrix, 'L')
    image.save(path)


def crop_all():
    for key, path in training_set.items():
        for filename in os.listdir(path):
            fullpath = os.path.join(path, filename)
            image = Image.open(fullpath).convert('L')
            np_image = numpy.array(image)
            np_image = binarize_array(np_image, 150)
            image = Image.fromarray(numpy.uint8(np_image), 'L')
            image = trim(image)
            image = image.resize((28, 28), Image.ANTIALIAS)
        image.save(fullpath)

def crop_image(path):
    image = Image.open(path).convert('L')
    np_image = numpy.array(image)
    np_image = binarize_array(np_image, 150)
    image = Image.fromarray(numpy.uint8(np_image), 'L')
    image = trim(image)
    image = image.resize((28, 28), Image.LANCZOS)
    image.save('/home/user/1.jpg', 'JPEG', quality=100)


if __name__ == '__main__':
    # filter('/home/user/1.jpg')
    crop_image('/home/user/img_92.jpg')
