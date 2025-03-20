import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import base64
import pathlib

# Set the path of the input folder

dataset = "https://drive.google.com/uc?export=download&id=1IXx1TynW-WmUlNNbJEvnExXDrmDvQlT3"
directory = tf.keras.utils.get_file('caras', origin=dataset, untar=True)
data = pathlib.Path(directory)

#print(folders)

# Import the images and resize them to a 128*128 size
# Also generate the corresponding labels

labels = []
images = []
listPersons = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']

size = 64,64
print('folders')
#folders.remove("LICENSE.txt")
print(listPersons)

for personName in listPersons:
    rostrosPath = os.path.join(data, personName)
    for fileName in os.listdir(rostrosPath):
        img_path = os.path.join(rostrosPath, fileName)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150))
        images.append(img)
        labels.append(personName)



# Transform the image array to a numpy type

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

# Develop a sequential model using tensorflow keras
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(listPersons), activation='softmax'))


# Compute the model parameters

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(-1, 150, 150, 1), y_train, epochs=50, validation_data=(X_test.reshape(-1, 150, 150, 1), y_test))

export_path = 'faces-model/1/'
tf.saved_model.save(model, os.path.join('./',export_path))
