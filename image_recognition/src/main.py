from limfocit import Limfocit_B
import os


if __name__ == '__main__':
	Limfocit_B.init(28, 28)
	# imgen = Limfocit_B.generate(250, 50, 50)
	# imgen.convert_to_image().show()
	image_path = '/home/user/research/Diploma/Diplom/image_recognition/images/trainingSet/2/'
	limfocit1 = Limfocit_B.get_from_image(os.path.join(image_path, 'img_56.jpg'), 0.4)
	limfocit2 = Limfocit_B.get_from_image(os.path.join(image_path, 'img_97.jpg'), 0.4)
	limfocit3 = Limfocit_B.get_from_image(os.path.join(image_path, 'img_55.jpg'), 0.4)
	limfocit4 = Limfocit_B.get_from_image(os.path.join(image_path, 'img_199.jpg'), 0.4)
	limfocit5 = Limfocit_B.get_from_image(os.path.join(image_path, 'img_235.jpg'), 0.4)


	affinity_list = []
	i = 0
	for filename in os.listdir(image_path):
		antigen = Limfocit_B.get_from_image(os.path.join(image_path, filename), 0.4)
		affinity_list.append(limfocit1.affinity(antigen))

		i += 1
		if i == 100:
			break


	pass