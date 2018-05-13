from limfocit import Limfocit_B
import os
import random



image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'

def load_limfocits(image_path):
    id_dict = {'0': [], '1': [], '2': [], '3': [], '4': [],
               '5': [], '6': [], '7': [], '8': [], '9': [] }
    for i in range(10):
        for filename in os.listdir(os.path.join(image_path, str(i)))[10:35]:
        # antigen = Limfocit_B.get_from_image(os.path.join(image_path, filename), 0.4)
            path = os.path.join(image_path, str(i), filename)
            limfo = Limfocit_B.get_from_image(path, 0.4)
            id_dict[str(i)].append(limfo)
    return id_dict

def antigen_reconition(antigen, id_dict):
    affinity_array = []
    max_id = -1
    max_affinity = -1

    for key, limfocites in id_dict.items():
        for limfo in limfocites:
            # aff = (limfo.affinity(antigen) + antigen.affinity(limfo))/2
            aff = limfo.affinity(antigen)
            if aff > max_affinity:
                max_id = limfo.id
                max_affinity = aff
            # affinity_array.append((limfo.id, limfo.affinity(antigen)))
    for key, limfocites in id_dict.items():
        for limfo in limfocites:
            if max_id == limfo.id:
                return (key, max_id, max_affinity)


# def antigen_reconition(antigen, id_dict):
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
#             # affinity_array[int(key)] += aff
#             if limfo.is_reacting(aff):
#                 affinity_array[int(key)] += 1
#
#     return affinity_array.index(max(affinity_array))


# def antigen_reconition(antigen, id_dict):
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
#             affinity_array[int(key)] += aff
#
#     return affinity_array.index(max(affinity_array))


def get_limfocit_by_id(id, id_dict):
    for key, limfocites in id_dict.items():
        for limfo in limfocites:
            if id == limfo.id:
                return limfo

# def load_antigens(image_path):
#     antigens = []
#     for i in range(10):
#         path = random.choice(os.listdir(os.path.join(image_path, str(i))))
#         antigen = Limfocit_B.get_from_image(os.path.join(image_path, str(i), path), 0.4)
#         antigen.value = i
#         antigens.append(antigen)
#     return antigens


def load_antigens(image_path, symbol):
    antigens = []
    for path in os.listdir(os.path.join(image_path, str(symbol))):
        antigen = Limfocit_B.get_from_image(os.path.join(image_path, str(symbol), path), 0.4)
        antigen.value = int(symbol)
        antigens.append(antigen)
    return antigens

def load_antigens_by_count(image_path, symbol, count):
    antigens = []
    for path in random.sample(os.listdir(os.path.join(image_path, str(symbol))), count):
        antigen = Limfocit_B.get_from_image(os.path.join(image_path, str(symbol), path), 0.4)
        antigen.value = int(symbol)
        antigens.append(antigen)
    return antigens

def run_reconition():
    image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'
    Limfocit_B.init()
    id_dict = load_limfocits(image_path)

    antigens = []
    values = []
    true_count = 0
    all_count = 0
    for symbol in range(10):
        antigens.append(load_antigens(image_path, symbol))

    for symbol in range(10):
        count = 0
        for a in antigens[symbol]:
            value = antigen_reconition(a, id_dict)
            # print(a.value, antigen_reconition(a, id_dict))
            if value == symbol:
                count += 1

        values.append((symbol, count / len(antigens[symbol])))
        true_count += count
        all_count += len(antigens[symbol])

    print(all_count, true_count)
    print(true_count * 100 / all_count, '%')
    print(values)


def reconition_symbol(symbol, count):
    image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'
    Limfocit_B.init()
    id_dict = load_limfocits(image_path)
    antigens = load_antigens_by_count(image_path, symbol, count)

    count = 0
    for a in antigens:
        # value = antigen_reconition(a, id_dict)
        key, max_id, max_affinity = antigen_reconition(a, id_dict)
        if int(key) == symbol:
            count += 1
    print(count * 100 / len(antigens), '%')




def main():
    reconition_symbol(0, 100)
    # image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'
    # Limfocit_B.init()
    # antigens = load_antigens_by_count(image_path, 5, 50)
    # limfocit = antigens[0]
    # for antigen in antigens:
    #     limfocit.mutation(antigen)
    #
    # limfocit.convert_to_image().show()



