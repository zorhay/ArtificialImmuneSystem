from PIL import Image
from scale import trim
import os
import numpy
from grey_scale import binarize_array, binarize_image

image_path = '/home/user/research/Diploma/ArtificialImmuneSystem/ImageRecognition/images/finally/'
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
    img = Image.open(path)
    members = [(0, 0)] * 9
    width, height = img.size
    newimg = Image.new("L", (width, height), "black")
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            members[0] = img.getpixel((i - 1, j - 1))
            members[1] = img.getpixel((i - 1, j))
            members[2] = img.getpixel((i - 1, j + 1))
            members[3] = img.getpixel((i, j - 1))
            members[4] = img.getpixel((i, j))
            members[5] = img.getpixel((i, j + 1))
            members[6] = img.getpixel((i + 1, j - 1))
            members[7] = img.getpixel((i + 1, j))
            members[8] = img.getpixel((i + 1, j + 1))
            members.sort()
            newimg.putpixel((i, j), (members[4]))
    newimg.save(path, 'JPEG')


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


def filter_all():
    for key, path in training_set.items():
        for filename in os.listdir(path):
            fullpath = os.path.join(path, filename)
            filter(fullpath)


if __name__ == '__main__':
    filter_all()
    # filter('/home/user/1.jpg')
    # crop_image('/home/user/img_92.jpg')
