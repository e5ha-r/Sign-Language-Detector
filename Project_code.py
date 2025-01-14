import cv2
import tensorflow as tf  
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Video capture setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Loading the trained model
classifier = Classifier(
    "keras_model.h5", 
    "labels.txt")

# Parameters
offset = 25
imgSize = 400
counter = 0

# Gesture trained labels
labels = ["Hello", "Thankyou", "Yes"]

try:
    # Main loop for video capturing and detection
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create a white background image for better prediction
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop the region of interest around the detected hand
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            # Adjust the aspect ratio of the cropped image
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Make predictions using the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

            # Display prediction text on the image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Display the cropped and resized images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        # Show the final output image with the prediction
        cv2.imshow('Image', imgOutput)
        cv2.waitKey(1)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()