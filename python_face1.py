# Fun script to recognise faces and draw squares around them
# INSTALL steps:
# pip3 install opencv-python module_name

import cv2
import os
import module_name

# Load image
img = cv2.imread('image.jpg')

# Path to your cv installation. Hack it in a better way if you want
module_path = os.path.dirname(module_name.__file__)
module_dir = os.path.dirname(module_path)
print(module_dir)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(module_dir+'/cv2/data/haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
