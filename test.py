import cv2
import os
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True, dest ='image', help='path to query image')
args = ap.parse_args()

# for i in os.listdir('Images'):
image = cv2.imread(args.image)
image = cv2.resize(image,(500,500))
# Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Blur 
gray = cv2.GaussianBlur(gray, (5,5),0)
gray = cv2.medianBlur(gray,5)
# Adaptive Thresh and edge detection
thresh = cv2.Canny(gray,50,150)
kernel = np.ones((3,3),np.uint8)

thresh = cv2.dilate(thresh,kernel,iterations=1)

cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)
for n,cnt in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    new_image = cv2.rectangle(image.copy(),(x,y),(x+w,y+h),(255,0,0),1)
    i = new_image[y:y+h,x:x+w]
    # cv2.imshow('i',new_image)
    # cv2.waitKey()
    if n >3:break


# cv2.imshow('ori',image)
# cv2.imshow('thresh',thresh)
# cv2.imshow('gray',gray)
# cv2.waitKey()

cnt = max(cnts, key = cv2.contourArea)
peri = cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

tr = tuple(approx[0][0])
tl = tuple(approx[1][0])
bl = tuple(approx[2][0])
br = tuple(approx[3][0])

# Perform Perspective Warping
src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(src_pts, dst)
warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# Draw Points
cv2.circle(image, tl, 0, color=(255,0,0), thickness=5)
cv2.circle(image, tr, 0, color=(0,255,0), thickness=5)
cv2.circle(image, bl, 0, color=(0,0,255), thickness=5)
cv2.circle(image, br, 0, color=(0,255,255), thickness=5)
# Display result
cv2.imshow('warp',warp)
cv2.imshow('ori',image)
cv2.imshow('thresh',thresh)
cv2.waitKey()
