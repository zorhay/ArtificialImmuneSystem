import random
from itertools import product


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

    def __init__(self, reaction_limit, limfocit_id):
        self.reaction_limit = reaction_limit
        self.id = limfocit_id
        self.id_list.append(limfocit_id)
        self.last_activity_time = None
        self.white_pixels = []
        self.black_pixels = []

    def affinity(self, antigen):
        p = 0
        for i in antigen.white_pixels:
            if i in self.white_pixels:
                 p += 1
        affinity = p / (len(self.white_pixels) + len(self.black_pixels))
        return affinity
            
    def is_reacting(self, affinity):
        return affinity >= self.reaction_limit

    def add_white_pixel(self, pixel):
        if pixel not in self.white_pixels:
            self.white_pixels.append(pixel)

    def add_black_pixel(self, pixel):
        if pixel not in self.black_pixels:
            self.black_pixels.append(pixel)

    @classmethod
    def generate(cls, limfocit_type, heigth=28, width=28):
        size = heigth * width
        white_indexes = set(random.sample(range(size), 8))
        black_indexes = set(range(size)) - white_indexes
        lim_id = Limfocit_B.get_id()
        print(lim_id)
        limfocit = Limfocit_B(reaction_limit=0.4, limfocit_id=lim_id)
        for i in white_indexes:
            limfocit.white_pixels.append(cls.limfoset[i])
        for i in black_indexes:
            limfocit.black_pixels.append(cls.limfoset[i])
        return limfocit


    @classmethod
    def get_id(cls):
        return cls.id_list[-1] + 1

    @classmethod
    def init(cls, limfoset_heigth=28, limfoset_width=28):
        cls.limfoset.clear()
        cls.id_list.clear()
        cls.limfoset = list(product(range(limfoset_heigth), range(limfoset_width)))
        cls.id_list.append(0)

    def __repr__(self):
        return "Limfocit: \n\
        reaction limit: {} \n\
        id: {} \n\
        last activity time: {} \n\
        white pixels: {} \n\
        black pixels: {} \n\
        ".format(self.reaction_limit, self.id, self.last_activity_time, self.white_pixels, self.black_pixels)