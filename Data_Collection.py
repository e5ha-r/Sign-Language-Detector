import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Video capture setup
cap = cv2.VideoCapture(0)  # 0 for device camera, use 1 for external webcam
detector = HandDetector(maxHands=1)  # Only one hand will be detected at a time
offset = 25
imgSize = 400
counter = 0

# Folder for saving collected data
folder_path = "C:/Users/STE/OneDrive/Desktop/Sign Language Detector Python/DATA/Yes"

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']  # Bounding box coordinates

            # Create a white background image for better reading by the machine
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]), max(0, x-offset):min(x+w+offset, img.shape[1])]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)  # Press key to capture the image
        if key == ord('s'):
            counter += 1
            cv2.imwrite(f'{folder_path}/Image_{time.time()}.jpg', imgWhite)
            print(counter)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()