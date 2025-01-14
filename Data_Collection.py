import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)  # 0 because camera is device camera, if you;re using web camera put 1 instead of 0
detector = HandDetector(maxHands = 1) # 1 beacuse only one hand will be detected at one time
offset = 25
imgSize = 400
counter = 0

Folder = "C:/Users/STE/OneDrive/Desktop/Sign Language Detector Pyhton/DATA/Yes"

if not os.path.exists(Folder):  #to check if the folder exists or not
    os.makedirs(Folder)

while True :
    success , img = cap.read ()
    hands , img = detector.findHands(img)
    if hands :
        hand = hands [0]
        x , y , w , h = hand ['bbox'] #all three axis and height

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 #image means the image bg wil be changed to white for better reading by the machine
        #3: Refers to the 3 color channels (RGB).
        #np.uint8: Sets the data type to 8-bit unsigned integers (0-255).
        #imgCrop = img[y-offset : y + h + offset , x-offset : x + w + offset]
        imgCrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]), max(0, x-offset):min(x+w+offset, img.shape[1])]
        imgCropShape = imgCrop.shape
        aspectratio = h/w

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop , (wCal , imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)            #maths ceil to round off the number to the smallest integer nearest to it
            imgWhite[: , wGap : wCal + wGap] = imgResize
        else :
            k = imgSize / w #w is weight
            hCal = math.ceil(k*h )  
            imgResize = cv2.resize(imgCrop , (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2) 
            imgWhite[hGap : hCal + hGap , :] = imgResize
        
        cv2.imshow('ImageCrop' , imgCrop)
        cv2.imshow('ImageWhite' , imgWhite)
    cv2.imshow('Image' , img)
    key = cv2.waitKey(1)   #key that would be pressed to start the camera
    if key == ord('s') :
        counter += 1
        cv2.imwrite(f'{Folder}/Image_{time.time()}.jpg' , imgWhite) # type: ignore
        print (counter)