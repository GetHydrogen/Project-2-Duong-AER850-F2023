#%% Import packages
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Step 1: Data Processing

shape = (100,100,3)
    
script_path = os.path.realpath(__file__)
root = os.path.dirname(script_path)
train_dir = os.path.join(root,'Data','Train')
validation_dir = os.path.join(root,'Data','Validation')

train_aug = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical'
)

validation_gen = validation_aug.flow_from_directory(
    validation_dir,
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical'
)
#%% Step 2: Neural Network Architecture Design
#   Step 3: Hyperparameter Analysis

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#%% Step 4: Model Evaluation

#Fit and save model
history = model.fit(train_gen, epochs=5, validation_data=(validation_gen))
model.save('saved_model_Duong_65857.h5')

#Plot Accuracy
plt.title('Step 4: Accuracy plot')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Step 4 Accuracy plot',dpi=200)
plt.show()

#Plot Loss
plt.title('Step 4: Loss plot')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Step 4 Loss plot',dpi=200)
plt.show()
