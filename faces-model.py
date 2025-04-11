import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import pathlib
import shutil

# Descargar y descomprimir el dataset desde Google Drive
dataset_url = "https://drive.google.com/uc?export=download&id=1ThUklHdF9_LhujOFvg6-2kxqB79vnO92"
directory = tf.keras.utils.get_file('losmasguaposdeltec', origin=dataset_url, untar=True)
data = pathlib.Path(directory)

# Lista de carpetas/personas esperadas
listPersons = ['Angel', 'Arkel', 'gabriel', 'Guadalupe', 'JULIO']

# Cargar imágenes y etiquetas
labels = []
images = []

print("Carpetas detectadas:")
print(listPersons)

for personName in listPersons:
    rostrosPath = os.path.join(data, personName)
    for fileName in os.listdir(rostrosPath):
        img_path = os.path.join(rostrosPath, fileName)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (150, 150))
            images.append(img)
            labels.append(personName)
        else:
            print(f"[WARNING] No se pudo leer la imagen: {img_path}")

# Convertir a numpy array y normalizar
images = np.array(images).astype('float32') / 255.0
labels = np.array(labels)

# Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42
)

# Dar forma para Keras (agregar canal de grises)
X_train = X_train.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)

# Definir el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(listPersons), activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Guardar el modelo
export_path = 'faces-model/1/'
if os.path.exists(export_path):
    shutil.rmtree(export_path)  # Eliminar si ya existe para evitar conflictos
tf.saved_model.save(model, export_path)

# Mostrar etiquetas codificadas
print("Clases:", label_encoder.classes_)
print("Etiquetas codificadas:", np.unique(labels_encoded))
