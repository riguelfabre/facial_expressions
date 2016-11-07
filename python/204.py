
# coding: utf-8

# In[2]:

import time 
import requests
import cv2
import operator
import pandas as pd
import numpy as np
from __future__ import print_function

# Import library to display results
import matplotlib.pyplot as plt
from matplotlib import pyplot
get_ipython().magic(u'matplotlib inline')
# Display images within Jupyter


# In[3]:

emotions = pd.read_csv('500_images.csv')


# In[4]:

filenames = emotions["Image Name"]


# In[7]:

#Scaling

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    height, width = img.shape[:2]
    res = cv2.resize(img,(10*width, 10*height), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('Scale'+str(num)+'.jpg', res)
    num=num+1


# In[8]:

#Translation

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    rows,cols = img.shape[:2]
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('Translation'+str(num)+'.jpg', dst)
    num=num+1


# In[9]:

#Rotation

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('Rotation'+str(num)+'.jpg', dst)
    num=num+1


# In[11]:

#Affine Transformation

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    rows,cols = img.shape[:2]
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('Affine'+str(num)+'.jpg', dst)
    num=num+1


# In[16]:

#Perspective Transformation

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    rows,cols = img.shape[:2]
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(300,300))
    cv2.imwrite('Perspective'+str(num)+'.jpg', dst)
    num=num+1


# In[28]:

#Gaussian Filtering

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name)
    Gblur = cv2.GaussianBlur(img,(5,5),3)
    cv2.imwrite('Gblur'+str(num)+'.jpg', Gblur)
    num=num+1


# In[6]:

#Erosion

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name,0)
    kernel = np.ones((10,10),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    cv2.imwrite('erosion'+str(num)+'.jpg', erosion)
    num=num+1


# In[7]:

#Dilation

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name,0)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    cv2.imwrite('Dilation'+str(num)+'.jpg', dilation)
    num=num+1


# In[8]:

#Gradient

num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name,0)
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite('Gradient'+str(num)+'.jpg', gradient)
    num=num+1


# In[11]:

#Sobel Derivatives


num = 1
for name in filenames:
    pathToFileInDisk = name
    img = cv2.imread(name,0)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    cv2.imwrite('Sobelx'+str(num)+'.jpg', sobelx)
    num=num+1

