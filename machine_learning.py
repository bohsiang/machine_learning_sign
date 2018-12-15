# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:41:32 2018

@author: user

"""
#lib---------------------------------------------------------------
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
'''
from keras import backend as K
K.set_image_dim_ordering('th')

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
'''
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os 
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#var-------------------------------------------------------------- 
img_rows,img_cols = 200,200  #image size

batch_size =  60  #批次訓練大小

nb_classes = 4      #classes 種類

nb_epoch = 50     #訓練週期

img_channels = 1    #num of channels

nb_filters = 32     #convelutional filter to use

nb_pool = 2         #size of pooling area for max pooling

nb_conv = 3         #convolution kernel size
#data-------------------------------------------------------------
path1 = 'D:\Desktop\machine_data\input_data'
path2 = 'D:\Desktop\machine_data\input_data_resized'

listing = os.listdir(path1)
num_samples=size(listing)   #path1有幾張照片
print (num_samples)
 
for file in listing:
     im = Image.open(path1 + '\\' + file)
     img = im.resize((img_rows,img_cols))
     gray = img.convert('L')            #################
     
     gray.save(path2+'\\' + file,"JPEG")
     
imlist = os.listdir(path2)

im1 = array(Image.open(path2+'\\'+ imlist[0])) # open index 為0的照片
m,n = im1.shape[0:2] #size 圖片為x*x
imnbr = len(imlist) #總共有多少張

immatrix = array([array(Image.open(path2+'\\'+im2)).flatten()for im2 in imlist],'f') #將每張圖片變成一維陣列

label = np.ones((num_samples,),dtype = int) #############
label[0:400]=0
label[400:835]=1
label[835:1507]=2
label[1507:]=3


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[167].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
print(train_data[0].shape)  #data size
print(train_data[1].shape)  #label size
#train set------------------------------------------------------------------------

(x,y)=(train_data[0],train_data[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=4) 

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train_shape',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
#plt.imshow(x_train[i, :, :,0], interpolation='nearest')
print("label:", Y_train[i,:])
# set model--------------------------------------------------------------------


model = Sequential()

model.add(Conv2D(filters=nb_filters,kernel_size=(nb_conv, nb_conv),input_shape=(img_rows,img_cols,1),activation='relu', padding='valid'))
#model.add(Dropout(0.3))
#model.add(Conv2D(filters=nb_filters,kernel_size=(nb_conv, nb_conv),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu', padding='valid'))
#model.add(Dropout(0.3))
#model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Conv2D(filters=128, kernel_size=(3, 3),activation='relu', padding='valid'))
#model.add(Dropout(0.3))
#model.add(Conv2D(filters=128, kernel_size=(3, 3),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
'''
model.add(Conv2D(filters=256, kernel_size=(3, 3),activation='relu', padding='valid'))
#model.add(Dropout(0.3))
#model.add(Conv2D(filters=128, kernel_size=(3, 3),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
'''
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary() 
'''
try:
    model = load_model('D:\Desktop\machine_data\savemodel\model.h5')
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")
'''
#model.summary()
startTime = time.time() 
train_history=model.fit(x_train, Y_train,validation_split=0.2,epochs= nb_epoch, batch_size=batch_size, verbose=1)  
takeTimes=time.time()  - startTime 
takeTimes=takeTimes/60
print("Time:"+str(takeTimes))
 
'''
score = model.evaluate(x_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(x_test[1:5]))
print(Y_test[1:5])
'''



import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


model.save('D:\Desktop\machine_data\savemodel\model1.h5',overwrite=True)
print("save model to disk")

#prediction--------------------------------------------------------------------
prediction=model.predict_classes(x_test)
prediction[:10]

label_dict={0:"20",1:"right",2:"stop",3:"traffic light"}

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+",predict="+str(prediction[idx]),fontsize=10) 
        
        ax.set_xticks([]);
        ax.set_yticks([])        
        idx+=1 
    plt.show()

x_test_normal=x_test.reshape(x_test.shape[0], img_rows, img_cols)
plot_images_labels_prediction(x_test_normal,y_test,prediction,0)

Predicted_Probability=model.predict(x_test)
print(Predicted_Probability)
























