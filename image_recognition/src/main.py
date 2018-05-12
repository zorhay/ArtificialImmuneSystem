from limfocit import Limfocit_B
import os
import random

id_dict = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': [],
        '8': [],
        '9': []
    }

image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'

def load_limfocits():
    global id_dict, image_path
    Limfocit_B.init(28, 28)
    for i in range(10):
        for filename in os.listdir(os.path.join(image_path, str(i)))[10:55]:
        # antigen = Limfocit_B.get_from_image(os.path.join(image_path, filename), 0.4)
            path = os.path.join(image_path, str(i), filename)
            limfo = Limfocit_B.get_from_image(path, 0.4)
            id_dict[str(i)].append(limfo)

# def antigen_reconition(antigen):
#     global id_dict
#     affinity_array = []
#     max_id = -1
#     max_affinity = -1
#
#     for key, limfocites in id_dict.items():
#         for limfo in limfocites:
#             # aff = (limfo.affinity(antigen) + antigen.affinity(limfo))/2
#             aff = limfo.affinity(antigen)
#             if aff > max_affinity:
#                 max_id = limfo.id
#                 max_affinity = aff
#             # affinity_array.append((limfo.id, limfo.affinity(antigen)))
#     for key, limfocites in id_dict.items():
#         for limfo in limfocites:
#             if max_id == limfo.id:
#                 return (key, max_id, max_affinity)


# def antigen_reconition(antigen):
#     global id_dict
#     affinity_array = []
#     for i in range(10):
#         affinity_array.append(0)
#
#     max_id = -1
#     max_affinity = -1
#
#     for key, limfocites in id_dict.items():
#         for limfo in limfocites:
#             aff = (limfo.affinity(antigen) + antigen.affinity(limfo))/2
#             # aff = limfo.affinity(antigen)
#             if limfo.is_reacting(aff):
#                 affinity_array[int(key)] += 1
#
#     return affinity_array.index(max(affinity_array))


def antigen_reconition(antigen):
    global id_dict
    affinity_array = []
    for i in range(10):
        affinity_array.append(0)

    max_id = -1
    max_affinity = -1

    for key, limfocites in id_dict.items():
        for limfo in limfocites:
            aff = (limfo.affinity(antigen) + antigen.affinity(limfo))/2
            # aff = limfo.affinity(antigen)
            affinity_array[int(key)] += aff

    return affinity_array.index(max(affinity_array))


def get_limfocit_by_id(id):
    for key, limfocites in id_dict.items():
        for limfo in limfocites:
            if id == limfo.id:
                return limfo

def main():
    global id_dict, image_path
    load_limfocits()
    antigens = []
    for i in range(10):
        path = random.choice(os.listdir(os.path.join(image_path, str(i))))
        antigen = Limfocit_B.get_from_image(os.path.join(image_path, str(i), path), 0.4)
        antigen.value = i
        antigens.append(antigen)

    for a in antigens:
        print(a.value, antigen_reconition(a))

    pass



if __name__ == '__main__':
    main()




















    # affinity_list = []
    # i = 0
    # for filename in os.listdir(image_path):
    #     antigen = Limfocit_B.get_from_image(os.path.join(image_path, filename), 0.4)
    #     aff = max((limfocit1.affinity(antigen) + antigen.affinity(limfocit1))/2,
    #         (limfocit2.affinity(antigen) + antigen.affinity(limfocit2))/2,
    #         (limfocit3.affinity(antigen) + antigen.affinity(limfocit3))/2,
    #         (limfocit4.affinity(antigen) + antigen.affinity(limfocit4))/2,
    #         (limfocit5.affinity(antigen) + antigen.affinity(limfocit5))/2,
    #         )
    #     affinity_list.append(aff)

    #     i += 1
    #     if i == 100:
    #         break

    # print([a for a in affinity_list if a > 0.4].__len__())
