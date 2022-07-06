#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:46:17 2022

@author: ali
"""
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from network.resnet_tf import * 
#import tensorflow_datasets as tfds

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(tf.__version__)
def train(batch_size,
          img_height,
          img_width,
          train_data_dir,
          val_data_dir,
          SAVE_MODEL_PATH,
          EPOCHS):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      train_data_dir,
      validation_split=0.1,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      val_data_dir,
      validation_split=0.1,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    
    class_names = train_ds.class_names
    print(class_names)
    
    
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    
    
    for image_batch, labels_batch in train_ds:
      print(image_batch.shape)
      print(labels_batch.shape)
      break
    
    image_batch, label_batch = next(iter(train_ds))
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].numpy().astype("uint8"))
      label = label_batch[i]
      plt.title(class_names[label])
      plt.axis("off")
    
    
    
    num_classes = 8
    
    model = resnet18()
    #model.build(input_shape=(None,32,32,3))
    #model.summary()
    '''
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])
    '''
    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    
    
    #epochs = 10
    
    history = model.fit(train_ds,
      validation_data=val_ds,
      epochs=EPOCHS
    )
    
    model.save(SAVE_MODEL_PATH)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(EPOCHS)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
if __name__=="__main__":
    net = "resnet"
    batch_size = 300
    img_height = 32
    img_width = 32
    train_data_dir = "/home/ali/repVGG/datasets/8/roi"
    val_data_dir = "/home/ali/repVGG/datasets/8/roi-test"
    SAVE_MODEL_PATH = "/home/ali/repVGG/datasets/model/" + net + ".pb"
    EPOCHS = 1
    
    
    train(batch_size,
          img_height,
          img_width,
          train_data_dir,
          val_data_dir,
          SAVE_MODEL_PATH,
          EPOCHS)
    