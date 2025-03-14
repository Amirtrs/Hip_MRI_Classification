# -*- coding: utf-8 -*-

import os
import cv2
import gdown
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
!pip install tensorflow
!pip install keras

from google.colab import drive
drive.mount('/content/gdrive')

from zipfile import ZipFile
zip_file_path = '/content/gdrive/My Drive/RadImageNet - Hip MRI.zip'

try:
    with ZipFile(zip_file_path, 'r') as zipObj:
        zipObj.extractall('/content/')
    print("File extracted successfully")
except zipfile.BadZipFile:
    print("Error in extracting file")


from google.colab import drive
drive.mount('/content/gdrive')
!ls '/content/RadImageNet - Hip MRI/'

main_dir = '/content/RadImageNet - Hip MRI'
classes = ['normal', 'osseous_lesion']

output_dir = 'DATA'
train_ratio = 0.7
test_and_val_ratio = 0.15
image_size = (224, 224)


os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for cls in classes:

    images = os.listdir(os.path.join(main_dir, cls))

    train_images, test_and_val_images = train_test_split(images, train_size=train_ratio)


    val_images, test_images = train_test_split(test_and_val_images, train_size=0.5)


    os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)


    for img in train_images:

        img_path = os.path.join(main_dir, cls, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        image = cv2.resize(image, image_size)


        image = clahe.apply(image)


        save_path = os.path.join(output_dir, 'train', cls, img)
        cv2.imwrite(save_path, image)


    for img in val_images:

        img_path = os.path.join(main_dir, cls, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        image = cv2.resize(image, image_size)

        # Apply CLAHE
        image = clahe.apply(image)


        save_path = os.path.join(output_dir, 'val', cls, img)
        cv2.imwrite(save_path, image)


    for img in test_images:

        img_path = os.path.join(main_dir, cls, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        image = cv2.resize(image, image_size)


        image = clahe.apply(image)


        save_path = os.path.join(output_dir, 'test', cls, img)
        cv2.imwrite(save_path, image)

print("Data has been split, processed, and copied")

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

data_dir = '/content/DATA'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

img_size = 224
batch_size = 150
epochs = 30


learning_rate = 0.0001


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary')


base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))


x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


optimizer = Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=epochs)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

loss, accuracy = model.evaluate(test_generator)

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype("int32")
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()