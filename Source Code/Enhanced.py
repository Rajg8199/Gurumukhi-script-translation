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


# === PRE-PROCESSING
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

# === SEGMENTATION

bw_img = grayscale_img<0.9
plt.imshow(bw_img)
plt.title('SEGMENTED IMAGE')
plt.show()

# === CONTOUR DETECTION

# read image
img_1 = np.double(bw_img)

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
ret,thresh = cv2.threshold(gray,0,1,0)

# get contours
contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# take the first contour
cnt = contours[0]

# compute the bounding rectangle of the contour
x,y,w,h = cv2.boundingRect(cnt)

# draw contour
img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

# draw the bounding rectangle
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# display the image with bounding rectangle drawn on it
cv2.imshow("Bounding Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === ROTATION AND LBP

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
    return label2rgb(mask, image=image, bg_label=0, alpha
