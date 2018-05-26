import random
from itertools import product
import numpy
from PIL import Image
from grey_scale import binarize_array


class Limphocit_B():
    """
    ab = [(a1, b1),(a2, b2),(a3, b3),(a4, b4)]
    (an, bn) - coordinates of white pixels
    cd = [(c1, d1),(c2, d2),(c3, d3),(c4, d4)]
    (cn, dn) - coordinates of black pixels
    S = S1 + S2
    S1 = len(ab)
    S2 = len(cd)
    """
    limfoset = []
    id_list = []

    def __init__(self, reaction_limit, size):
        self.reaction_limit = reaction_limit
        self.id = self.id_list[-1] + 1
        self.id_list.append(self.id)
        self.last_activity_time = None
        self.size = size
        self.white_pixels = []
        self.black_pixels = []

    def affinity(self, antigen):
        p = 0
        for i in antigen.white_pixels:
            if i in self.white_pixels:
                 p += 1
        affinity_1 = p / (len(self.white_pixels))
        p = 0
        for i in self.white_pixels:
            if i in antigen.white_pixels:
                 p += 1
        affinity_2 = p / (len(antigen.white_pixels))
        return min(affinity_1, affinity_2)

    def alter_affinity(self, antigen):
        p = 0
        for i in antigen.white_pixels:
            if i in self.white_pixels:
                p += 1
        affinity = p / (len(self.white_pixels))  # + len(self.black_pixels))
        return affinity

    def is_reacting(self, affinity):
        return affinity >= self.reaction_limit

    def add_white_pixel(self, pixel):
        if pixel not in self.white_pixels:
            self.white_pixels.append(pixel)

    def add_black_pixel(self, pixel):
        if pixel not in self.black_pixels:
            self.black_pixels.append(pixel)

    def convert_to_array(self):
        img_array = []
        line = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.white_pixels:
                    line.append(255)
                else:
                    line.append(0)
            img_array.append(line)
            line = []
        return img_array

    def convert_to_image(self):
        ''' Creates PIL image '''
        return Image.fromarray(numpy.uint8(self.convert_to_array()), 'L')

    # def mutation(self, antigen):
    #     mutation_power = 0.05
    #     input_pixel_count = (len(antigen.white_pixels)**2*mutation_power)/len(self.white_pixels)
    #     input_pixel_count = int(input_pixel_count)
    #     output_pixel_count = (len(self.white_pixels) **2*mutation_power)/len(antigen.white_pixels)
    #     output_pixel_count = int(output_pixel_count)
    #
    #     if len(self.white_pixels) > output_pixel_count:
    #         output_pixels = set(random.sample(self.white_pixels, output_pixel_count))
    #     else:
    #         output_pixels = set(self.white_pixels)
    #     self.black_pixels = list(set(self.black_pixels) | output_pixels)
    #
    #     if len(antigen.white_pixels) > input_pixel_count:
    #         input_pixels = set(random.sample(antigen.white_pixels, input_pixel_count))
    #     else:
    #         input_pixels = set(antigen.white_pixels)
    #     for pixel in input_pixels:
    #         if pixel in self.black_pixels:
    #             self.black_pixels.remove(pixel)
    #     self.white_pixels = list(set(self.white_pixels) | input_pixels)

    def mutation(self, antigen, mutation_power=0.1):
        div_antigen_pixels = set([pixel for pixel in antigen.white_pixels if pixel not in self.white_pixels])
        pixel_count = int(len(div_antigen_pixels)*mutation_power)
        input_pixels = set(random.sample(div_antigen_pixels, pixel_count))
        self.white_pixels = list(set(self.white_pixels) | input_pixels)

        div_limphocit_pixels = set([pixel for pixel in self.white_pixels if pixel not in antigen.white_pixels])
        pixel_count = int(len(div_limphocit_pixels) * mutation_power)
        output_pixels = set(random.sample(div_limphocit_pixels, pixel_count))
        self.black_pixels = list(set(self.black_pixels) | output_pixels)

    def filter(self):
        img = self.convert_to_image()
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

        self.white_pixels.clear()
        self.black_pixels.clear()
        image_array = numpy.uint8(newimg)
        image_array.setflags(write=1)
        image_binarize_array = binarize_array(image_array)
        # TODO resize image
        # TODO crop image
        # TODO resize image again
        for i in range(image_binarize_array.shape[0]):
            for j in range(image_binarize_array.shape[1]):
                if image_binarize_array[i][j] == 255:
                    self.add_white_pixel((i, j))
                elif image_binarize_array[i][j] == 0:
                    self.add_black_pixel((i, j))
                else:
                    raise ValueError('image not binarized')

    @classmethod
    def generate(cls, limphocit_type, heigth=28, width=28):
        size = heigth * width
        white_indexes = set(random.sample(range(size), limphocit_type))
        black_indexes = set(range(size)) - white_indexes
        limphocit = cls(reaction_limit=0.4, size=heigth)
        for i in white_indexes:
            limphocit.white_pixels.append(cls.limfoset[i])
        for i in black_indexes:
            limphocit.black_pixels.append(cls.limfoset[i])
        return limphocit

    @classmethod
    def init(cls, limfoset_heigth=28, limfoset_width=28):
        cls.limfoset.clear()
        cls.id_list.clear()
        cls.limfoset = list(product(range(limfoset_heigth), range(limfoset_width)))
        cls.id_list.append(0)

    @classmethod
    def get_from_image(cls, image_path, reaction_limit=0.4):
        image = Image.open(image_path).convert('L')
        image_array = numpy.uint8(image)
        image_array.setflags(write=1)
        image_binarize_array = binarize_array(image_array)
        limphocit = cls(reaction_limit=reaction_limit, size=image.size[0])
        for i in range(image_binarize_array.shape[0]):
            for j in range(image_binarize_array.shape[1]):
                if image_binarize_array[i][j] == 255:
                    limphocit.add_white_pixel((i, j))
                elif image_binarize_array[i][j] == 0:
                    limphocit.add_black_pixel((i, j))
                else:
                    raise ValueError('image not binarized')
        return limphocit

    @classmethod
    def get_from_pil_image(cls, pil_image, reaction_limit=0.4):
        image_array = numpy.uint8(pil_image)
        image_array.setflags(write=1)
        image_binarize_array = binarize_array(image_array)
        # TODO resize image
        # TODO crop image
        # TODO resize image again
        limphocit = cls(reaction_limit=reaction_limit, size=pil_image.size[0])
        for i in range(image_binarize_array.shape[0]):
            for j in range(image_binarize_array.shape[1]):
                if image_binarize_array[i][j] == 255:
                    limphocit.add_white_pixel((i, j))
                elif image_binarize_array[i][j] == 0:
                    limphocit.add_black_pixel((i, j))
                else:
                    raise ValueError('image not binarized')
        return limphocit

    def __repr__(self):
        return "Limphocit: \n\
        reaction limit: {} \n\
        id: {} \n\
        last activity time: {} \n\
        white pixels: {} \n\
        black pixels: {} \n\
        ".format(self.reaction_limit, self.id, self.last_activity_time, self.white_pixels, self.black_pixels)