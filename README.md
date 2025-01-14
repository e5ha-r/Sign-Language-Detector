# **Sign-Language-Detector**
This project involves a Sign Language Detection system that uses computer vision and machine learning to detect and classify hand gestures representing specific signs. The system uses a webcam to capture real-time video and performs hand detection to recognize gestures from a pre-trained model.

**Features:**
1. Hand Detection: Uses the cvzone HandTrackingModule to detect and track hand gestures in real-time.
2. Gesture Classification: Classifies gestures using a pre-trained TensorFlow model. Supported gestures include "Hello", "Thankyou", and "Yes".
3. Data Collection: The system captures images of the hand gestures and saves them in a specified folder for later training purposes.
4. Real-time Prediction: Once a gesture is detected, it is classified using the loaded machine learning model and displayed on the screen with the corresponding label.

**Requirements:**
Python 3.x
OpenCV (cv2)
TensorFlow
cvzone HandTrackingModule
Numpy
Keras

**File Structure:**
1. Data Collection Script: Collects images of hand gestures for training.
2. Gesture Classification Script: Classifies real-time hand gestures using the pre-trained model.
3. Model Files: The keras_model.h5 file (the trained model) and labels.txt file (the corresponding gesture labels).