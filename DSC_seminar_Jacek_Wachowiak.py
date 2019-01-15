# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:42:58 2018
in Anaconda Spyder Python 3.6 with opencv
@author: Jacek Wachowiak
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#load image named 1.jpg results are named after the algorithm
img = cv2.imread('1.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=gray #not to deal with colors, can be commented out

#SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
#keypoints
img2=cv2.drawKeypoints(gray, kp,1)
cv2.imwrite('_SIFTsimple.jpg',img2)
#keypoints with size and orientation
img3=cv2.drawKeypoints(gray, kp, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('_SIFTcomplex.jpg',img3)
#descriptor
sift = cv2.xfeatures2d.SIFT_create()
kpS, desS = sift.detectAndCompute(gray,None)

#FAST
fast = cv2.FastFeatureDetector_create()
#keypoints
kpF = fast.detect(img,None)
img4 = cv2.drawKeypoints(img, kpF, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#all default params
print ("FAST parameters:")
print ("Threshold: ", fast.getThreshold())
print ("nonmaxSuppression: ", fast.getNonmaxSuppression())
print ("neighborhood: ", fast.getType())
print ("Total Keypoints with nonmaxSuppression: ", len(kpF))
cv2.imwrite('_FASTwithsuppression.jpg',img4)
#disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kpF2 = fast.detect(img,None)
print ("Total Keypoints without nonmaxSuppression: ", len(kpF2))
img5 = cv2.drawKeypoints(img, kpF2, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('_FASTwithoutsuppression.jpg',img5)

#BRIEF
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#keypoints
kp = star.detect(img, None)
#descriptors
kpB, desB = brief.compute(img, kp)
print ("BRIEF - Size of the descriptor:",desB.shape)
img6 = cv2.drawKeypoints(img, kpB, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('_BRIEF.jpg',img6)

#ORB
orb = cv2.ORB_create()
#keypoints and descriptors
kpO, desO = orb.detectAndCompute(img, None)
img7 = cv2.drawKeypoints(img, kpO, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('_ORB.jpg',img7)

#brute-force feature matching for ORB
#x=object searched
imgX = cv2.imread('x.jpg')
x= cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
kpX, desX = orb.detectAndCompute(x, None)
#create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#match descriptors.
matches = bf.match(desX,desO)
#sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
#draw first 15 matches.
imgResult = cv2.drawMatches(x, kpX, img, kpO, matches[:15], img, flags=4)
cv2.imwrite('_imgResultORB.jpg', imgResult)

#brute-force feature matching for SIFT
#x2=object searched
imgX = cv2.imread('x.jpg')
x2= cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
kpX2, desX2 = sift.detectAndCompute(x2,None)
#create BFMatcher object
bf = cv2.BFMatcher()
#matches
matches = bf.knnMatch(desX2,desS, k=2)
good = []
for m,n in matches:
    if m.distance < 0.2*n.distance: #trial and error selected to be visible, default=0.75 produces too many
        good.append([m])
#draw first 15 matches.
imgResult2 = cv2.drawMatchesKnn(x2, kpX2, img, kpS, good, img, flags=4)
cv2.imwrite('_imgResultSIFT.jpg', imgResult2)