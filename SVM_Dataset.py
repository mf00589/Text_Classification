import os
import io
import pickle
import requests
import zipfile
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from imageio import imread, imsave
from skimage import img_as_float, color, exposure
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib.cm as cm
import matplotlib.pyplot as plt


path_a = './training_type/a/'
path_b = './training_type/b/'
path_c = './training_type/c/'
path_d = './training_type/d/'
path_e = './training_type/e/'
path_f = './training_type/f/'
path_g = './training_type/g/'
path_h = './training_type/h/'
path_i = './training_type/i/'
path_j = './training_type/j/'
path_k = './training_type/k/'
path_l = './training_type/l/'
path_m = './training_type/m/'
path_n = './training_type/n/'
path_o = './training_type/o/'
path_p = './training_type/p/'
path_q = './training_type/q/'
path_r = './training_type/r/'
path_s = './training_type/s/'
path_t = './training_type/t/'
path_u = './training_type/u/'
path_v = './training_type/v/'
path_w = './training_type/w/'
path_x = './training_type/x/'
path_y = './training_type/y/'
path_z = './training_type/z/'
path_0 = './training_type/0/'
path_1 = './training_type/1/'
path_2 = './training_type/2/'
path_3 = './training_type/3/'
path_4 = './training_type/4/'
path_5 = './training_type/5/'
path_6 = './training_type/6/'
path_7 = './training_type/7/'
path_8 = './training_type/8/'
path_9 = './training_type/9/'
path_a1 = './ocr/alphabet/letters/a1/'
path_a2 = './ocr/alphabet/letters/a2/'
path_a3 = './ocr/alphabet/letters/a3/'
path_a4 = './ocr/alphabet/letters/a4/'
path_a5 = './ocr/alphabet/letters/a5/'
path_a6 = './ocr/alphabet/letters/a6/'
path_a7 = './ocr/alphabet/letters/a7/'
path_a8 = './ocr/alphabet/letters/a8/'

a_filenames = sorted([filename for filename in os.listdir(path_a) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
b_filenames = sorted([filename for filename in os.listdir(path_b) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
c_filenames = sorted([filename for filename in os.listdir(path_c) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
d_filenames = sorted([filename for filename in os.listdir(path_d) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
e_filenames = sorted([filename for filename in os.listdir(path_e) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
f_filenames = sorted([filename for filename in os.listdir(path_f) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
g_filenames = sorted([filename for filename in os.listdir(path_g) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
h_filenames = sorted([filename for filename in os.listdir(path_h) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
i_filenames = sorted([filename for filename in os.listdir(path_i) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
j_filenames = sorted([filename for filename in os.listdir(path_j) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
k_filenames = sorted([filename for filename in os.listdir(path_k) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
l_filenames = sorted([filename for filename in os.listdir(path_l) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
m_filenames = sorted([filename for filename in os.listdir(path_m) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
n_filenames = sorted([filename for filename in os.listdir(path_n) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
o_filenames = sorted([filename for filename in os.listdir(path_o) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
p_filenames = sorted([filename for filename in os.listdir(path_p) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
q_filenames = sorted([filename for filename in os.listdir(path_q) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
r_filenames = sorted([filename for filename in os.listdir(path_r) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
s_filenames = sorted([filename for filename in os.listdir(path_s) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
t_filenames = sorted([filename for filename in os.listdir(path_t) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
u_filenames = sorted([filename for filename in os.listdir(path_u) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
v_filenames = sorted([filename for filename in os.listdir(path_v) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
w_filenames = sorted([filename for filename in os.listdir(path_w) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
x_filenames = sorted([filename for filename in os.listdir(path_x) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
y_filenames = sorted([filename for filename in os.listdir(path_y) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
z_filenames = sorted([filename for filename in os.listdir(path_z) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
one_filenames = sorted([filename for filename in os.listdir(path_0) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
two_filenames = sorted([filename for filename in os.listdir(path_2) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
three_filenames = sorted([filename for filename in os.listdir(path_3) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
four_filenames = sorted([filename for filename in os.listdir(path_4) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
five_filenames = sorted([filename for filename in os.listdir(path_5) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
six_filenames = sorted([filename for filename in os.listdir(path_6) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
seven_filenames = sorted([filename for filename in os.listdir(path_7) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
eight_filenames = sorted([filename for filename in os.listdir(path_8) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
nine_filenames = sorted([filename for filename in os.listdir(path_9) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
aone_filenames = sorted([filename for filename in os.listdir(path_a1) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
atwo_filenames = sorted([filename for filename in os.listdir(path_a2) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
athree_filenames = sorted([filename for filename in os.listdir(path_a3) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
afour_filenames = sorted([filename for filename in os.listdir(path_a4) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
afive_filenames = sorted([filename for filename in os.listdir(path_a5) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
asix_filenames = sorted([filename for filename in os.listdir(path_a6) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
aseven_filenames = sorted([filename for filename in os.listdir(path_a7) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
aeigth_filenames = sorted([filename for filename in os.listdir(path_a8) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])



data = []
labels = []
a_filenames = [path_a+filename for filename in a_filenames]
b_filenames = [path_b+filename for filename in b_filenames]
c_filenames = [path_c+filename for filename in c_filenames]
d_filenames = [path_d+filename for filename in d_filenames]
e_filenames = [path_e+filename for filename in e_filenames]
f_filenames = [path_f+filename for filename in f_filenames]
g_filenames = [path_g+filename for filename in g_filenames]
h_filenames = [path_h+filename for filename in h_filenames]
i_filenames = [path_i+filename for filename in i_filenames]
j_filenames = [path_j+filename for filename in j_filenames]
k_filenames = [path_k+filename for filename in k_filenames]
l_filenames = [path_l+filename for filename in l_filenames]
m_filenames = [path_m+filename for filename in m_filenames]
n_filenames = [path_n+filename for filename in n_filenames]
o_filenames = [path_o+filename for filename in o_filenames]
p_filenames = [path_p+filename for filename in p_filenames]
q_filenames = [path_q+filename for filename in q_filenames]
r_filenames = [path_r+filename for filename in r_filenames]
s_filenames = [path_s+filename for filename in s_filenames]
t_filenames = [path_t+filename for filename in t_filenames]
u_filenames = [path_u+filename for filename in u_filenames]
v_filenames = [path_v+filename for filename in v_filenames]
w_filenames = [path_w+filename for filename in w_filenames]
x_filenames = [path_x+filename for filename in x_filenames]
y_filenames = [path_y+filename for filename in y_filenames]
z_filenames = [path_z+filename for filename in z_filenames]
one_filenames = [path_1+filename for filename in one_filenames]
two_filenames = [path_2+filename for filename in two_filenames]
three_filenames = [path_3+filename for filename in three_filenames]
four_filenames = [path_4+filename for filename in four_filenames]
five_filenames = [path_5+filename for filename in five_filenames]
six_filenames = [path_6+filename for filename in six_filenames]
seven_filenames = [path_7+filename for filename in seven_filenames]
eight_filenames = [path_8+filename for filename in eight_filenames]
nine_filenames = [path_9+filename for filename in nine_filenames]
aone_filenames = [path_a1+filename for filename in aone_filenames]
atwo_filenames = [path_a2+filename for filename in atwo_filenames]
athree_filenames = [path_a3+filename for filename in athree_filenames]
afour_filenames = [path_a4+filename for filename in afour_filenames]
afive_filenames = [path_a5+filename for filename in afive_filenames]
asix_filenames = [path_a6+filename for filename in asix_filenames]
aseven_filenames = [path_a7+filename for filename in aseven_filenames]
aeight_filenames = [path_a8+filename for filename in aeigth_filenames]



for filename in a_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(0)
print('Finished adding A samples to dataset')


for filename in b_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(1)
print('Finished adding B samples to dataset')

for filename in c_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(2)
print('Finished adding C samples to dataset')


for filename in d_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(3)
print('Finished adding D samples to dataset')




for filename in e_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(4)
print('Finished adding E samples to dataset')

for filename in f_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(5)
print('Finished adding F samples to dataset')

for filename in g_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(6)
print('Finished adding G samples to dataset')

for filename in h_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(7)
print('Finished adding H samples to dataset')

for filename in i_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(8)
print('Finished adding I samples to dataset')

for filename in j_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(9)
print('Finished adding J samples to dataset')

for filename in k_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(10)
print('Finished adding K samples to dataset')

for filename in l_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(11)
print('Finished adding L samples to dataset')

for filename in m_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(12)
print('Finished adding M samples to dataset')

for filename in n_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(13)
print('Finished adding N samples to dataset')

for filename in o_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(14)
print('Finished adding O samples to dataset')

for filename in p_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(15)
print('Finished adding P samples to dataset')

for filename in q_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(16)
print('Finished adding Q samples to dataset')

for filename in r_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(17)
print('Finished adding R samples to dataset')

for filename in s_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(18)
print('Finished adding S samples to dataset')


for filename in t_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(19)
print('Finished adding T samples to dataset')

for filename in u_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(20)
print('Finished adding U samples to dataset')

for filename in v_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(21)
print('Finished adding V samples to dataset')

for filename in w_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(22)
print('Finished adding W samples to dataset')

for filename in x_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(23)
print('Finished adding X samples to dataset')

for filename in y_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(24)
print('Finished adding Y samples to dataset')

for filename in z_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(25)
print('Finished adding Z samples to dataset')

for filename in one_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(26)
print('Finished adding 1 samples to dataset')

for filename in two_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(27)
print('Finished adding 2 samples to dataset')

for filename in three_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(28)
print('Finished adding 3 samples to dataset')

for filename in four_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(29)
print('Finished adding 4 samples to dataset')

for filename in five_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(30)
print('Finished adding 5 samples to dataset')

for filename in six_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(31)
print('Finished adding 6 samples to dataset')

for filename in seven_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(32)
print('Finished adding 7 samples to dataset')

for filename in eight_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(33)
print('Finished adding 8 samples to dataset')

for filename in nine_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(34)
print('Finished adding 9 samples to dataset')

for filename in aone_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(35)
print('Finished adding a1 samples to dataset')

for filename in atwo_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(36)
print('Finished adding a2 samples to dataset')

for filename in athree_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(37)
print('Finished adding a3 samples to dataset')

for filename in afour_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(38)
print('Finished adding a4 samples to dataset')

for filename in afive_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(39)
print('Finished adding a5 samples to dataset')

for filename in asix_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(40)
print('Finished adding a6 samples to dataset')

for filename in aseven_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(41)
print('Finished adding a7 samples to dataset')

for filename in aeight_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    data.append(hog_features)
    labels.append(42)
print('Finished adding a8 samples to dataset')

print('Training the SVM')
#create the SVC
clf = LinearSVC(dual=False,verbose=1)
#train the svm
clf.fit(data, labels)

pickle.dump( clf, open( "place.detector", "wb" ) )

































