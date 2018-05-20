#
#
# def limfocit_to_image(limfocit):
#     img_array = []
#     line = []
#     for i in range(28):
#         for j in range(28):
#             if (i, j) in limfocit.white_pixels:
#                 line.append(True)
#             else:
#                 line.append(False)
#         img_array.append(line)
#         line = []
#     return img_array



###### FILTER ######

# import sys
# import numpy as np
# from PIL import Image
# from grey_scale import binarize_image
#
#
# path = '/home/user/img_1.jpg'
#
# def MyFilter(Box):
#     temp_arr = [0 for i in range(9)]
#     counter = 0;
#     for v in range(3):
#         for h in range(3):
#             temp_arr[counter] = Box[v][h]
#             counter = counter + 1
#     temp_arr = np.sort(temp_arr)
#     return temp_arr[4]
#
#
# Image.fromarray(binarize_image(path, None, 160), 'L').save(path)
#
# img = Image.open(path)
# img1 = np.array(img, dtype=np.uint8)
# Matrix = [[0 for x in range(img.size[0])] for y in range(img.size[1])]
#
# for i in range(img.size[0] - 2):
#     for y in range(img.size[1] - 2):
#         local = img1[i:i + 3, y:y + 3]
#         Matrix[i + 1][y + 1] = MyFilter(local)
#
# # finish part
# image_matrix = np.uint8(Matrix)
# image = Image.fromarray(image_matrix, 'L').save('clearImage.png', 'PNG')

