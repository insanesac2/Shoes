# USAGE
# python sliding_window.py --image images/adrian_florida.jpg 

# import the necessary packages
import numpy as np 
import cv2
from scipy.misc import imresize
from feature_extractor import feature
import pickle
from PIL import Image


image = cv2.imread(r'D:\Profiledskin\test\17.png')
image = imresize(image, 0.5)
row,col,ch = image.shape
a = c = d =[] 


b, g, r = cv2.split(image)

(winW, winH) = (3,3)
stepSize=1
windowSize=(winW, winH)
for y in xrange(0, r.shape[0]-winH+1):
		for x in xrange(0, r.shape[1]-winW+1):
			a.append((image[y:y + windowSize[1], x:x + windowSize[0]]))
                
l = len(a)     

arr = []
vec = np.zeros((l,15))
pred = np.zeros(l)  

for i in range(0,l):
    vec[i] = feature(a[i])

filename = r'D:\Profiledskin\shoes\finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))    

pred = (clf.predict(vec))  
mat = np.reshape(pred, (-1,col-2))

mat[mat == 1] = 255


mat = mat.astype('uint8')

thresh = cv2.threshold(mat,127,255,0)[1]
im = mat
contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
idx =0 
c = max(contours, key = cv2.contourArea)
area = cv2.contourArea(c)
area = area/2
for cnt in contours:
    if (cv2.contourArea(cnt)>area):
      x,y,w,h = cv2.boundingRect(cnt)
      cv2.rectangle(mat,(x,y),(x+w,y+h),(200,200,0),2)
mat = Image.fromarray(mat)
mat.show()
