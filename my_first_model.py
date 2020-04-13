#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os


# In[2]:


img_width=150
img_height=150


# In[3]:


from tensorflow.keras import backend as K

if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
print(input_shape)


# In[4]:


#path="C:\\Users\\rexdi\\Desktop\\flowers\\"

train_dir='./Train'
valid_dir='./Test'

epochs=10
batch_size=100


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Image agumentation
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) 
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 


# In[6]:


train_generator = train_datagen.flow_from_directory(train_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical') 
  
validation_generator = test_datagen.flow_from_directory( 
                                    valid_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical') 


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,Activation
from tensorflow.keras.optimizers import Adam


# In[8]:


model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='softmax'))
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(metrics=['accuracy'],optimizer='adam',loss='categorical_crossentropy')

model.summary()


# In[9]:


train_generator.classes.shape,validation_generator.classes.shape


# In[ ]:


op = model.fit_generator(train_generator,epochs=48,validation_data=validation_generator)


# In[ ]:


model.save('op.h5')


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(op.history['loss'])
plt.plot(op.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(op.history['accuracy'])
plt.plot(op.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array
#Data Load
flower_test=load_img(path+'\\b_rose.jpg',target_size=(150,150))
#print(flower_test.format)

flower_img_array=img_to_array(flower_test)
print('Conversion to array',flower_img_array.shape)

flower = np.expand_dims(flower_img_array, axis=0)         #Pre processing
print('Dimension expansion',flower.shape)

pred=model.predict(flower)
op_flower=np.argmax(pred,axis=1)

if op_flower==0:
    print('Daisy')
elif op_flower==1:
    print('Dandelion')
elif op_flower==2:
    print('Rose')
elif op_flower==3:
    print('Sunflower')
elif op_flower==4:
    print('Tulip')

