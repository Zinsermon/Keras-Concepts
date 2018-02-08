from preprocessing import *
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM,LeakyReLU,BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.losses import binary_crossentropy
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical


data, labels = GetForexData()
## To Applying All losses Not Just sparse
labels = to_categorical(labels, num_classes=2)
model = Sequential()


model.add(Dense(100,input_shape=(97,),activation='relu'))
## Normalizing the weights after relu because relu just split the even it may too high to exploid gradients
## accuracy increase from 50 to 70
BatchNormalization(axis=1)
model.add(Dense(100,activation='softmax'))
model.add(Dense(100,activation='softmax'))
model.add(Dense(100,activation='relu'))
## Normalizing the weights after relu because relu just split the even it may too high to exploid gradients
## accuracy increase from 50 to 70
BatchNormalization(axis=1)
model.add(Dense(100,activation='relu'))
## Normalizing the weights after relu because relu just split the even it may too high to exploid gradients
## accuracy increase from 50 to 70
BatchNormalization(axis=1)
model.add(Dense(2,activation='softmax'))
model.compile(Adam(lr=0.001),loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
model.fit(x=data,y=labels,batch_size=2239,epochs=1000,validation_split=0.1)

print('\n\n\n')
##model.summary()
