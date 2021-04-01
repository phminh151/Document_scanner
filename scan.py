import cv2
import os
import numpy as np

for i in os.listdir('test'):
    image = cv2.imread('test/'+i)
    # image = cv2.resize(image,(500,500))
    # Grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Blur 
    gray = cv2.GaussianBlur(gray, (7,7),0)
    gray = cv2.medianBlur(gray,5)
    # Adaptive Thresh and edge detection
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,3)
    # cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = cv2.Canny(gray,50,150)
    kernel = np.ones((3,3),np.uint8)
    edge = cv2.dilate(edge,kernel,iterations=1)
    # for row in range(edge.shape[0]):
    # # Find the left-most and right-most pixel in the sqare
    #     start, stop = 0, 0
    #     for col in range(edge.shape[1]):
    #         # Find the left-most
    #         if edge[row,col] != 0 and start == 0: start = col, row
    #         # Find the right-most
    #         if edge[row,col] != 0: stop = col, row 
    #     # If there was a pixel in that row, connect them with a line
    #     if start != 0:
    #         cv2.line(edge, start, stop, 255, 1)
    try:
        # Find Biggest Contours
        cnts,_ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key = cv2.contourArea)

        # Find 4 Points for Perspective Warping
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
        # Save result
        print(i)
        cv2.imshow('warp.jpg',warp)
        cv2.imshow('thresh.jpg',edge)
        cv2.imshow('ori.jpg',image)
        # cv2.waitKey()
        
    except:
        continue