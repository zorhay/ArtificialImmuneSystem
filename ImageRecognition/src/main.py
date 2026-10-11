if __package__:
    from .immune_algorithm import DEFAULT_AFFINITY, ImmuneAlgorithm, image_path, training_set, limpho_path, limphocites_set
else:
    from immune_algorithm import DEFAULT_AFFINITY, ImmuneAlgorithm, image_path, training_set, limpho_path, limphocites_set

if __name__ == '__main__':
    im_alg = ImmuneAlgorithm(28, 28, DEFAULT_AFFINITY, 21)
    im_alg.reload_limphocites(limphocites_set)
