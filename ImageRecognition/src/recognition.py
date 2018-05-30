from immune_algorithm import ImmuneAlgorithm, limphocites_set
from limphocit import Limphocit_B
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please enter image path!')
        sys.exit(0)
    elif len(sys.argv) > 2:
        print('Many arguments are entered.')
        sys.exit(0)
    input_image = sys.argv[1]

    print('Immune System initialization...')
    im_alg = ImmuneAlgorithm(28, 28, 0.8, 21)
    print('Loading limphocites...')
    im_alg.reload_limphocites(limphocites_set)
    print('Recognition: ' + sys.argv[1] + ' ...')
    result = im_alg.recognition_image(input_image)
    print('result: ' + result)

