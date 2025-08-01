import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import os

human_name_path = 'human.json'

TRAIN_DIR = r'Faces/train/'

try:
    with open(human_name_path, 'r') as f:
        humans = json.load(f)
        humans = humans["human"]
    # print(humans)
except FileNotFoundError:
    print(f"Error: The file '{human_name_path}' was not found.")

haar_cascade = cv.CascadeClassifier('haar_face.xml')


# Training images
features=[]
# Labels
labels=[]

def preprocess_and_face_detection():
    for human in humans:
        path = os.path.join(TRAIN_DIR, human)
        label = humans.index(human)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)

            if img_array is None:
                continue

            img_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            img_array = cv.equalizeHist(img_array)

            # img_array = cv.bilateralFilter(img_array, 5, 175, 150)
            # median = cv.medianBlur(img_array, 3)
            # gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY)

            # detect faces in the image
            # apply multiple Haar cascades
            haar_cascades = [
                cv.CascadeClassifier('haar_face.xml'),
                cv.CascadeClassifier('haarcascade_frontalface_default.xml'),
                cv.CascadeClassifier('haarcascade_profileface.xml')
            ]

            faces_rect = []
            for cascade in haar_cascades:
                detected = cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5)
                for rect in detected:
                    if not any(np.array_equal(rect, existing_rect) for existing_rect in faces_rect):
                        faces_rect.append(rect)
            # faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5)

            for(x,y,w,h) in faces_rect:
                faces_roi = img_array[y:y+h, x:x+w] # crop faces to get the regions of interest
                features.append(faces_roi)
                labels.append(label)

preprocess_and_face_detection()
print('Preprocessing and Face detection done ------------')

print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)
