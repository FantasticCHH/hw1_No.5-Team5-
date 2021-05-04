#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import csv
#import matplotlib.pyplot as plt
from time import time
import os
import cv2
import numpy as np
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt


# In[2]:


num_classes = 8
channels=3
img_rows=64
img_cols=64
img_pixel=(img_rows*img_cols*channels)
trainEpochs =20#執行15次訓練週期
batchSize = 20 #每一批次比數100
photo_channel=39


# In[3]:


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(basePath, contains=None):
    # 返回有效的图片路径数据集
    return list_files(basePath, validExts=image_types, contains=contains)
 
def list_files(basePath, validExts=None, contains=None):
    # 遍历图片数据目录，生成每张图片的路径
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环遍历当前目录中的文件名
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
 
            # 通过确定.的位置，从而确定当前文件的文件扩展名
            ext = filename[filename.rfind("."):].lower()
 
            # 检查文件是否为图像，是否应进行处理
            if validExts is None or ext.endswith(validExts):
                # 构造图像路径
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


# In[4]:


print("------load data------")
path1 = 'C:/Users/eric/Desktop/ML/2HW/CIFAR10/'
data = []
labels = []

# 路徑
imagePaths = sorted(list(list_images(path1)))
random.seed(10)
random.shuffle(imagePaths)
 
# 讀取
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows,img_cols))
    data.append(image)
    
    # 建立label
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data,
     labels, test_size=0.2, random_state=42)

#將y_test、y_train轉換成(總數,標籤)，若無lb，則只會顯示(總數) 
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

print("------load success------")


# In[5]:


model = Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),
                  input_shape=(64,64,3),
                  activation='relu',
                  padding='same'))

model.add(Dropout(rate=0.25))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=64,kernel_size=(3,3),
                activation='relu',
                  padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))



model.add(Dense(y_train.shape[1],activation='softmax'))


# In[6]:


model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# H = model.fit(x=x_train,
#                           y=y_train,
#                           validation_split=0,
#                           epochs=trainEpochs,
#                           batch_size=batchSize,
#                           verbose=1)
H = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
              epochs=trainEpochs, batch_size=batchSize)


# In[7]:


from sklearn.metrics import classification_report
("[INFO] evaluating network...")
predictions = model.predict(x=x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, trainEpochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.savefig("C:/Users/eric/Desktop/ML/2HW/demo/")

plt.figure()
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (alexnet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("C:/Users/eric/Desktop/ML/2HW/demo/")

