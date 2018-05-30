from immune_algorithm import ImmuneAlgorithm, image_path, training_set, limpho_path, limphocites_set

if __name__ == '__main__':
    im_alg = ImmuneAlgorithm(28, 28, 0.4, 20)
    im_alg.reload_limphocites(limphocites_set)
