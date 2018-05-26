from immune_algorithm import ImmuneAlgorithm, image_path, training_set, limpho_path, limphocites_set, mutant_path, mutant_set




if __name__ == '__main__':
    im_alg = ImmuneAlgorithm(28, 28, 0.4, 20)
    im_alg.reload_limphocites(limphocites_set)
    print(im_alg.recognition_image(training_set['5']+'img_1371.jpg'))





























    # im_alg.training(training_set, 50)
    # im_alg.save_images(mutant_path)
    # im_alg.create_network(limphocites_set)




    # print(im_alg.test_recognition_symbol(training_set['5'], 50, '0'))
    # results = im_alg.test_recognition_all(training_set, 100)
    # print(results)

