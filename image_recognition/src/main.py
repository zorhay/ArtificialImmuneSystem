from immune_algorithm import ImmuneAlgorithm

image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/'
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

if __name__ == '__main__':
    im_alg = ImmuneAlgorithm(28, 28, 0.4, 20)
    im_alg.create_network(training_set)
    # print(im_alg.recognition_image(training_set['5']+'img_1371.jpg'))
    # print(im_alg.test_recognition_symbol(training_set['5'], 50, '0'))
    results = im_alg.test_recognition_all(training_set, 50)
    print(results)

