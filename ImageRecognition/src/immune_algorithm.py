from math import exp
import os
import random
from limphocit import Limphocit_B

image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'
limpho_path = '/home/user/research/Diploma/Diplom/image_recognition/images/limphocites/'
mutant_path = '/home/user/research/Diploma/Diplom/image_recognition/images/finally/'
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

limphocites_set = {
                '0': limpho_path + '0/',
                '1': limpho_path + '1/',
                '2': limpho_path + '2/',
                '3': limpho_path + '3/',
                '4': limpho_path + '4/',
                '5': limpho_path + '5/',
                '6': limpho_path + '6/',
                '7': limpho_path + '7/',
                '8': limpho_path + '8/',
                '9': limpho_path + '9/'
                }

mutant_set = {
                '0': mutant_path + '0/',
                '1': mutant_path + '1/',
                '2': mutant_path + '2/',
                '3': mutant_path + '3/',
                '4': mutant_path + '4/',
                '5': mutant_path + '5/',
                '6': mutant_path + '6/',
                '7': mutant_path + '7/',
                '8': mutant_path + '8/',
                '9': mutant_path + '9/'
                }


class ImmuneAlgorithm(object):
    counters = {} # affinityes
    immune_system = {} # limphocites
    main_affinity_to_react = None
    antigen_width = None
    antigen_height = None
    network_power = None
    training_set = {}

    def __init__(self, antigen_width, antigen_height, main_affinity_to_react, network_power):
        self.antigen_width = antigen_width
        self.antigen_height = antigen_height
        self.main_affinity_to_react = main_affinity_to_react
        self.network_power = network_power

    def get_counters(cls):
        return cls.counters

    def get_immune_system(cls):
        return cls.immune_system

    def set_affinity_to_react(cls, affinity):
        cls.main_affinity_to_react = affinity

    def get_increment_value(self, affinity):
        try:
            increment = affinity * exp(1 / (1 - affinity))
        except (OverflowError, ZeroDivisionError):
            increment = float('inf')
        return increment

    def clear_counters(self):
        for key in self.counters.keys():
            self.counters[key] = 0

    def save_immune_network_to_memory(cls, path):
        # TODO implement, use pickle
        pass

    def load_immune_network_from_memory(cls, path):
        #TODO  load immune network from memory
        pass

    def create_network(self, training_set):
        self.immune_system.clear()
        self.counters.clear()
        for key in training_set.keys():
            self.immune_system[key] = []
            self.counters[key] = 0
        Limphocit_B.init()
        for symbol, path in training_set.items():
            # for filename in os.listdir(path)[0:self.network_power]:
            for filename in random.sample(os.listdir(path), self.network_power):
                fullpath = os.path.join(path, filename)
                self.immune_system[symbol].append(Limphocit_B.get_from_image(fullpath))


    def training(self, training_set, set_size):
        for symbol, dirpath in training_set.items():
            for filename in os.listdir(dirpath)[:set_size]:
                fullpath = os.path.join(dirpath, filename)
                antigen = self.load_antigen_by_path(fullpath)
                for limphocit in self.immune_system[symbol]:
                    limphocit.mutation(antigen, 0.1)


    def reload_limphocites(self, training_set):
        self.immune_system.clear()
        self.counters.clear()
        for key in training_set.keys():
            self.immune_system[key] = []
            self.counters[key] = 0
        Limphocit_B.init()
        for symbol, path in training_set.items():
            for filename in os.listdir(path):
                fullpath = os.path.join(path, filename)
                self.immune_system[symbol].append(Limphocit_B.get_from_image(fullpath))

    def load_antigen_by_path(self, path):
        return Limphocit_B.get_from_image(path)

    def recognition_antigen(self, antigen):
        self.clear_counters()
        for key, limphocites in self.immune_system.items():
            for limphocit in limphocites:
                affinity = limphocit.affinity(antigen)
                if affinity >= self.main_affinity_to_react:
                    increment = self.get_increment_value(affinity)
                    self.counters[key] += increment
        return max(self.counters, key=self.counters.get)

    def recognition_image(self, path):
        antigen = Limphocit_B.get_from_image(path)
        # return self.recognition_antigen(antigen)
        return {'symbol': '5'}

    def test_recognition_all(self, training_set, antigen_count):
        results = {}
        for symbol, path in training_set.items():
            antigens = []
            truth_count = 0
            for filename in random.sample(os.listdir(path), antigen_count):
                antigen = Limphocit_B.get_from_image(os.path.join(path, filename))
                antigen.value = symbol
                antigens.append(antigen)
            for antigen in antigens:
                if self.recognition_antigen(antigen) == antigen.value:
                    truth_count += 1
            results[symbol] = truth_count/len(antigens)
        return results

    def test_recognition_symbol(self, symbol_dir, antigen_count, value):
        antigens = []
        results = []
        for i in range(10):
            results.append(0)
        truth_count = 0
        for filename in random.sample(os.listdir(symbol_dir), antigen_count):
            antigen = Limphocit_B.get_from_image(os.path.join(symbol_dir, filename))
            antigen.value = value
            antigens.append(antigen)
        for antigen in antigens:

            results[int(self.recognition_antigen(antigen))] += 1
            # if self.recognition_antigen(antigen) == antigen.value:
            #     truth_count += 1
        # return truth_count/len(antigens)
        return results


    def save_images(self, dir):
        for symbol, limphocites in self.immune_system.items():
            os.mkdir(os.path.join(dir, symbol), mode=0o777)
            for limphocit in limphocites:
                limphocit.convert_to_image().save(os.path.join(dir, symbol, str(limphocit.id)+'.jpg'))


