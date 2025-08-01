import numpy as np
import cv2 as cv

features=np.load('features.npy', allow_pickle=True)
labels=np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# To overcome repetition of above steps, we will save the trained model to load it anywhere
face_recognizer.save('face_trained.yml')