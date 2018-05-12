
# coding: utf-8

# In[2]:


import grey_scale
import scale
from PIL import Image
import numpy as np

def get_bin_array(bimg):
	return np.array(bimg)


# In[3]:


from limfocit import Limfocit_B


# In[46]:


imb = grey_scale.png_to_yuv('../../mnist_dataset/trainingSample/img_115.jpg')


# In[51]:


np.asarray(imb)[0][0]


# In[6]:


imb1 = grey_scale.png_to_yuv('/home/user/black.jpg')


# In[7]:


imb1 = scale.resize_image('/home/user/black.jpg', '/home/user/black_mini.jpg', 1)


# In[8]:


imb1 = grey_scale.png_to_yuv('/home/user/black_mini.jpg')


# In[9]:


imb1


# In[10]:


np.asarray(imb1)


# In[12]:





# In[13]:


imgen.black_pixels


# In[14]:


imarr = limfocit_to_image(imgen)


# In[15]:


imarr = np.asarray(imarr)


# In[16]:


imarr


# In[17]:


image = Image.fromarray(imarr, '1')
image.show()


# In[37]:


test = [] #np.array()
line = [] #np.array()
# for i in range(128):
#     for j in range(128):
#         line.append(True)
#     test.append(line)
test = []
test.append([True, False, True, True])
test.append([True, False, False, True])
test.append([True, True, True, True])
test.append([True, True, True, True])
im_arr = np.array(test)
im_test = Image.fromarray(im_arr, 'L')
im_test.save('/home/user/black_test.jpg')
# im_test.show()


# In[120]:


# imb1 = Image.open('/home/user/jesus.png', mode='r')
imb1 = grey_scale.png_to_yuv('/home/user/jesus.png')
# imb1 = Image.open('/home/user/jesus_yuv.jpg')


# In[121]:


# imb1.convert('1')
np.array(imb1, dtype=np.int32)


# In[91]:


i = Image.fromarray(im_arr, mode='L').convert('1')
# i.save('/home/user/im.jpg')
i.show()


# In[92]:


import numpy as np


#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV( rgb ):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv


# In[93]:


imb1 = np.array(Image.open('/home/user/jesus.png'))


# In[94]:


im = RGB2YUV(imb1)


# In[95]:


im


# In[103]:


a=Image.fromarray(im, mode='RGB').convert('1')


# In[104]:


np.array(a)


# In[127]:


from PIL import Image

def png_to_yuv(input_image, output_image=None):
    img = Image.open(input_image, mode="r")
    img_yuv = img.convert('L')
    point = calc_eta(img_yuv)
    bw = img_yuv.point(lambda x: 0 if x<point else 255, '1')
    if output_image:
        bw.save(output_image)
    return bw

def calc_eta(img):
    width,height = img.size
    total = 0
    for i in range(0,width):
        for j in range(0,height):
            total += img.getpixel((i,j))
    mean = total / (width * height)
    return mean


# In[132]:


imb1 = Image.open('/home/user/jesus.png', mode='r')
np.array(imb1)
# imb1 = Image.open('/home/user/jesus_yuv.jpg')


# In[140]:


input_image = '/home/user/jesus.png'
output_image = None

img = Image.open(input_image)
img_yuv = img.convert('1')
point = calc_eta(img_yuv)
print(point)
bw = img_yuv.point(lambda x: 0 if x<point else 255, '1')
if output_image:
    bw.save(output_image)
    
imb1 = bw


# In[141]:


np.array(imb1, dtype=np.int32)


# In[154]:


def limfocit_to_image(limfocit):
    img_array = []
    line = []
    for i in range(28):
        for j in range(28):
            if (i, j) in limfocit.white_pixels:
                line.append(True)
            else:
                line.append(False)
        img_array.append(line)
        line = []
    return img_array
                    


# In[143]:


from PIL import Image
import numpy


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


# In[146]:


img = binarize_image('/home/user/jesus.png', None, 200)


# In[149]:


i = Image.fromarray(img)


# In[151]:





# In[152]:


Limfocit_B.init()


# In[153]:


imgen = Limfocit_B.generate(200)


# In[156]:


arr = limfocit_to_image(imgen)


# In[163]:


k = np.array(arr)
i = Image.fromarray(k)


# In[158]:


i

