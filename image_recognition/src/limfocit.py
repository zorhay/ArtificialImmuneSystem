import random
from itertools import product
import numpy
from PIL import Image
from grey_scale import binarize_array


class Limfocit_B():
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
        affinity = p / ((len(self.white_pixels)) + len(self.black_pixels))
        return affinity

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

    def mutation(self, antigen):
        mutation_power = 0.05
        input_pixel_count = (len(antigen.white_pixels)**2*mutation_power)/len(self.white_pixels)
        input_pixel_count = int(input_pixel_count)
        output_pixel_count = (len(self.white_pixels) **2*mutation_power)/len(antigen.white_pixels)
        output_pixel_count = int(output_pixel_count)

        if len(self.white_pixels) > output_pixel_count:
            output_pixels = set(random.sample(self.white_pixels, output_pixel_count))
        else:
            output_pixels = set(self.white_pixels)
        self.black_pixels = list(set(self.black_pixels) | output_pixels)

        if len(antigen.white_pixels) > input_pixel_count:
            input_pixels = set(random.sample(antigen.white_pixels, input_pixel_count))
        else:
            input_pixels = set(antigen.white_pixels)
        for pixel in input_pixels:
            if pixel in self.black_pixels:
                self.black_pixels.remove(pixel)
        self.white_pixels = list(set(self.white_pixels) | input_pixels)

    def filter(self):
        # TODO implement this
        pass

    @classmethod
    def generate(cls, limfocit_type, heigth=28, width=28):
        size = heigth * width
        white_indexes = set(random.sample(range(size), limfocit_type))
        black_indexes = set(range(size)) - white_indexes
        limfocit = Limfocit_B(reaction_limit=0.4, size=heigth)
        for i in white_indexes:
            limfocit.white_pixels.append(cls.limfoset[i])
        for i in black_indexes:
            limfocit.black_pixels.append(cls.limfoset[i])
        return limfocit

    @classmethod
    def init(cls, limfoset_heigth=28, limfoset_width=28):
        cls.limfoset.clear()
        cls.id_list.clear()
        cls.limfoset = list(product(range(limfoset_heigth), range(limfoset_width)))
        cls.id_list.append(0)

    @classmethod
    def get_from_image(cls, image_path, reaction_limit):
        image = Image.open(image_path).convert('L')
        image_array = numpy.uint8(image)
        image_array.setflags(write=1)
        image_binarize_array = binarize_array(image_array)
        # TODO resize image
        # TODO crop image
        # TODO resize image again
        limfocit = cls(reaction_limit=reaction_limit, size=image.size[0])
        for i in range(image_binarize_array.shape[0]):
            for j in range(image_binarize_array.shape[1]):
                if image_binarize_array[i][j] == 255:
                    limfocit.add_white_pixel((i, j))
                elif image_binarize_array[i][j] == 0:
                    limfocit.add_black_pixel((i, j))
                else:
                    raise ValueError('image not binarized')
        return limfocit

    def __repr__(self):
        return "Limfocit: \n\
        reaction limit: {} \n\
        id: {} \n\
        last activity time: {} \n\
        white pixels: {} \n\
        black pixels: {} \n\
        ".format(self.reaction_limit, self.id, self.last_activity_time, self.white_pixels, self.black_pixels)