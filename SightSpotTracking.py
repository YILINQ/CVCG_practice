import json
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# 1. Collect face-eye image
# 2. Vectorize eye image and build filter
# 3. Build CNN to predict output

def create_model():
    model = Sequential()
    model.add(Convolution2D(
        # batch_input_shape=(64, 1, 28, 28),
        filters=20,
        kernel_size=5,
        strides=1,
        input_shape=[25, 50, 3],
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=[2, 2],
        strides=[2, 2],
    ))

    model.add(Flatten())
    model.add(Dropout(0.2))

    # 2 output x,y
    model.add(Dense(units=2, activation='tanh'))

    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='mean_squared_error', metrics=['accuracy'])

    return model


def eye_catch():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    while True:
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = img[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
        cv2.imshow("image", img)
        cv2.waitKey(1)


eye_catch()
