import cv2 as cv
import numpy as np
import json
import os

human_name_path = 'human.json'
VAL_TEST_DIR = r'Faces/val/'

try:
    with open(human_name_path, 'r') as f:
        humans = json.load(f)
        humans = humans["human"]
    # print(humans)
except FileNotFoundError:
    print(f"Error: The file '{human_name_path}' was not found.")

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# Display an image from features
import matplotlib.pyplot as plt

# Assuming features contains a list/array of face images (as numpy arrays)
sample_idx = 0  # Change index to display different images
sample_image = features[sample_idx]

plt.imshow(sample_image, cmap='gray')
plt.title(f"Sample Feature Image (Index {sample_idx})")
plt.axis('off')
plt.show()

haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

def recognition(img_path):
    img = cv.imread(img_path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    # cv.imshow('Person', gray)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {humans[label]} with a confidence of {confidence}')

        cv.putText(img, str(humans[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected Face', img)

# r'Faces/val/ben_afflek/1.jpg'
img_name = 'ben_afflek/4.jpg'
img_path = os.path.join(VAL_TEST_DIR, img_name)
recognition(img_path)

cv.waitKey(0)
