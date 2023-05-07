

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image

# === GETTING INPUT SIGNAL

import warnings
warnings.filterwarnings('ignore')
filename = askopenfilename()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2



img = mpimg.imread(filename)
plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()


# PRE-PROCESSING
h1=100
w1=100
dimension = (w1, h1) 
from skimage.transform import rescale, resize, downscale_local_mean
resized_image = resize(img,(h1,w1))

plt.imshow(resized_image)
plt.title('RESIZED IMAGE')
plt.show()

from skimage.color import rgb2gray

grayscale_img = rgb2gray(img)
bw_img = grayscale_img<0.9
plt.imshow(bw_img)
plt.title('SEGMENTED IMAGE')
plt.show()


import cv2
import numpy as np

# read image
img_1 = np.double(bw_img)

# convert to grayscale

# threshold

# get contours

# result = img.copy()


# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,1,0)



# contours = cv2.findContours(img_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# for cntr in contours:
#     x,y,w,h = cv2.boundingRect(cntr)
#     cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     print("x,y,w,h:",x,y,w,h)
 







# # import required libraries
# import cv2

# # read the input image
# img = resized_image

# # convert the image to grayscale
# gray = resized_image[:,:,0]

# # apply thresholding on the gray image to create a binary image
# ret,thresh = cv2.threshold(gray,0.7,1,0)

# # find the contours
# contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# # take the first contour
# cnt = contours[0]

# # compute the bounding rectangle of the contour
# x,y,w,h = cv2.boundingRect(cnt)

# # draw contour
# img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

# # draw the bounding rectangle
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# # display the image with bounding rectangle drawn on it
# cv2.imshow("Bounding Rectangle", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

# settings for LBP
radius = 3
n_points = 8 * radius

image = gray

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

METHOD = 'uniform'
image = gray

image = gray
lbp = local_binary_pattern(image, n_points, radius, METHOD)


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')
    
plt.show()
import pickle

with open('Trainfea1.pickle', 'rb') as fp:
     Train_features = pickle.load(fp)

Labels = np.arange(0,20)
Labels[0:4] = 1
Labels[4:8] = 2
Labels[8:12] = 3
Labels[12:16] = 4
Labels[16:20] = 5


# -- CNN
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

test_data1 = os.listdir('Dataset/1/')
test_data2 = os.listdir('Dataset/2/')
test_data3 = os.listdir('Dataset/3/')
test_data4 = os.listdir('Dataset/4/')
test_data5 = os.listdir('Dataset/5/')

dot= []
labels = []

for img in test_data1:
    
    try:
        img_1 = plt.imread('Dataset/1' + "/" + img)
        img_resize = cv2.resize(img_1,((100, 100)))
        dot.append(np.array(img_resize))
        labels.append(0)
        
    except:
        None
        
for img in test_data2:
    
    try:
        img_2 = plt.imread('Dataset/2'+ "/" + img)
        img_resize = cv2.resize(img_2,(100, 100))
        
        dot.append(np.array(img_resize))
        labels.append(1)
        
    except:
        None

for img in test_data3:
    
    try:
        img_2 = plt.imread('Dataset/3'+ "/" + img)
        img_resize = cv2.resize(img_2,(100, 100))
        
        dot.append(np.array(img_resize))
        labels.append(2)
        
    except:
        None

for img in test_data4:
    
    try:
        img_2 = plt.imread('Dataset/4'+ "/" + img)
        img_resize = cv2.resize(img_2,(100, 100))
        
        dot.append(np.array(img_resize))
        labels.append(3)
        
    except:
        None

for img in test_data5:
    
    try:
        img_2 = plt.imread('Dataset/5'+ "/" + img)
        img_resize = cv2.resize(img_2,(100, 100))
        
        dot.append(np.array(img_resize))
        labels.append(4)
        
    except:
        None        
        
from keras.utils import to_categorical
import os
import argparse
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout

#  ===========================================================================

# -- Splitting Train and Test data
        
x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.2, random_state = 101)

x_train1=np.zeros((len(x_train),100,100,3))

for i in range(0,len(x_train)):
        x_train1[i,:,:]=x_train[i]

x_test1=np.zeros((len(x_test),100,100,3))

for i in range(0,len(x_test)):
        x_test1[i,:,:]=x_test[i]     

Test_s = 20
Train_s = 80
print('==================================')
print('Percentage Of Test Data = ',Test_s)
print('Percentage Of Train Data = ',Train_s)
print('==================================')

#  ===========================================================================
#  CNN Model

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)

test_Y_one_hot = to_categorical(y_test)





# -- LSTM


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import  Dropout,Dense
import numpy as np

from sklearn import metrics

print("====================================================")
print("            Long Short Term Memory                  ")
print("====================================================")
print()

# === DIMENSION FITTING ===

x=dot
Y=labels

# === MODEL INITIALIZATION ===

model = Sequential()

# === INPUT LAYER ===

model.add(LSTM(input_shape=(20,1), kernel_initializer="uniform", return_sequences=True, stateful=False, units=50))
model.add(Dropout(0.2))

# === HIDDEN LAYER ===

model.add(LSTM(5, kernel_initializer="uniform", activation='relu',return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(3,kernel_initializer="uniform",activation='relu'))
model.add(Dense(1, activation='linear'))

# === COMPILATION ===

model.compile(loss="mae", optimizer='adam',metrics=['accuracy'])
model.summary()
print()

# # === FITTING ===

# model.fit(x, Y, batch_size = 100, epochs = 1)

# Evaluate=model.evaluate(x,Y,verbose=1)

# y_pred1 = model.predict(x)


# # ========================== PERFORMANCE ANALAYSIS ============================

# # ===== confusion matrix ======

# print("====================================================")
# print("             Performance Analysis for LSTM          ")
# print("====================================================")
# print()

# accuracy1=Evaluate[1]*100
# print("1. LSTM Accuracy:",accuracy1,'%')
# print()



#  ===========================================================================
# -- KNN

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Train_features, Labels)
Class_knn = neigh.predict([counts])

print('-----------------------------')
print('IMAGE to TEXT TRANSLATE')
if Class_knn == 1:
    print('-----------------------------')
    print('Respected Input Character')
    print('ਓ')
    print('Generated Output')
    print('o')
elif Class_knn == 2:
    print('2')
    print('A')
    
elif Class_knn == 3:
    print('ਏ')
    print('I')
    
elif Class_knn == 4:
    print('4')
    print('S')
    
elif Class_knn == 5:
    print('5')
    print('H')
    
print('-----------------------------')

test_data1_text = os.listdir('../Source Code/text/')
train_text_fea = []
for len_test in range(1,len(test_data1_text)+1):
    
    with open('text/'+str(len_test)+'.txt', encoding='utf8') as f:
        for line in f:
            print(line.strip())
            
    A = line.strip()
    
    temp_train = []
    for ii in range(0,len(A)):
        
        temp_train.append(ord(A[1]))
    train_text_fea.append(temp_train)

filename1 = askopenfilename()

with open(filename1, encoding='utf8') as f:
    for line in f:
        print(line.strip())
        
A1 = line.strip()
temp_test = []
for ii in range(0,len(A1)):
    
    temp_test.append(ord(A1[1]))

print('=======================')
print('Text Process')
Labels = np.arange(0,len(test_data1_text))
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_text_fea, Labels)
Class_knn = neigh.predict([temp_test])    

print('-----------------------------')
print('Text to English')
print('-----------------------------')
print('Recognized')
with open('eng/'+str(int(Class_knn)+1)+'.txt', encoding='utf8') as f:
    for line in f:
        print(line.strip())
print('-----------------------------')

# file1 = open("1.txt","r")
# print(file1.read(1))
# print()
# file1.close()

# from translate import Translator
# translator= Translator(to_lang="English")
# translation = translator.translate("ਓ")
# print(translation)


# from translate import Translator
# translator= Translator(to_lang="Tamil")
# translation = translator.translate('now')
# print(translation)

    