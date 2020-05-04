from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential

import numpy as np 

model=Sequential()

model.open('op.h5')
#Data Load
flower_test=load_img('sun.jpg',target_size=(150,150))
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