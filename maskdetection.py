# -*- coding: utf-8 -*-
"""maskDetection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SYL3hDrQTtj1fE3xhVZhpLa1kbuLVhVL
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
import seaborn as sns
from tensorflow.keras.models import save_model
from tensorflow.math import confusion_matrix

directory = '/home/shrivastavahs/MaskDetection/archive/data'
categories = ['with_mask','without_mask']

Image_size = 100
data = []
for category in categories:
    label = categories.index(category)
    folder = os.path.join(directory,category)
    for img_name in os.listdir(folder):
        image_path = os.path.join(folder,img_name)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image,(Image_size,Image_size))
            data.append([resized_image,label])

twoindices = []
has = False
nothave = False
index = 0
for feature, label in data:
    if has == False and label == 0:
        twoindices.append(index)
        has = True
    elif nothave == False and label == 1:
        twoindices.append(index)
        nothave = True
    if has == True and nothave == True:
        break
    index += 1

for index in twoindices:
    print('not masked' if data[index][1] == 1 else 'masked')
    plt.imshow(data[index][0])
    plt.show()

random.shuffle(data)

X = []
Y = []
for feature, label in data:
    X.append(feature)
    Y.append(label)

print(f'X:{len(X)}, Y:{len(Y)}')

Y = np.array(Y)
X = np.array(X)
print(X.shape)
print(Y.shape)

# Scaling images values
X = X/255

for i in range(5):
    print('masked' if Y[i] == 0 else 'not masked')
    plt.imshow(X[i])
    plt.show()

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.95,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)

Input_size = (Image_size,Image_size,3)
Num_classes = 2

Model = Sequential([
    Input(shape=Input_size),
    Conv2D(32,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),

    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(Num_classes,activation='softmax')
])

Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

result = Model.fit(x_train,y_train,validation_split=0.1,epochs=7)

evaluation = Model.evaluate(x_test,y_test)
print("the loss value is: ",evaluation[0])
print("the accuracy value is: ",evaluation[1])

plt.figure(figsize=(7,7))
plt.plot(result.history['accuracy'],color='red')
plt.plot(result.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','val_accuracy'],loc='lower right')

predicted_y = Model.predict(x_test)

y_predicted_values = []
for value in predicted_y:
    y_predicted_values.append(np.argmax(value))
comparison = []
for predicted_value,true_value in zip(y_predicted_values,y_test):
    comparison.append([predicted_value,true_value])
print(comparison)

print(len(comparison))

plt.figure(figsize=(5,5))
conf_matrix = confusion_matrix(y_test,y_predicted_values)
sns.heatmap(conf_matrix,square=True,cbar=True,annot=True,annot_kws={'size':8},cmap='Blues')

# def detect_mask(image_path):
#     image = cv2.imread(image_path)
#     if image.shape[-1] == 1:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     resized_image = cv2.resize(image,(Image_size,Image_size))
#     resized_image = resized_image / 255
#     prediction = Model.predict(np.expand_dims(resized_image, axis=0))
#     print(prediction)
#     predicted_class = ['Masked','Not Masked'][np.argmax(prediction)]
#     print(predicted_class)

# detect_mask('hima_with.jpeg')
# detect_mask('hima_without.jpeg')

def detect_mask(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        if image.shape[-1] == 1:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized_image = cv2.resize(image, (Image_size, Image_size))
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)  # Add batch dimension and scale
        prediction = Model.predict(resized_image)
        predicted_class = ['Masked', 'Not Masked'][np.argmax(prediction)]
        print(predicted_class)
    else:
        print("Error: Image not found or couldn't be read.")

detect_mask('hima_with.jpeg')
detect_mask('hima_without.jpeg')
Model.save('model.h5')



