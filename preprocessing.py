from sklearn.preprocessing import MinMaxScaler
import numpy as np
from random import randint
import pandas as pd

def GetForexData():
    data = pd.read_csv('D:\Yo World !\Keras-Concepts\dataset01_eurusd4h.csv',index_col = 0)
    labels = data['tipo']
    del data['tipo']
    #print(labels.head(1),"  ",labels.shape,' ',data.shape)
    data = np.array(data)
    labels = np.array(labels)
    data = np.delete(data,(0),axis=0)
    #print(data[0])
    labels = np.delete(labels,(0),axis=0)
    #print(labels,'',labels.size,'',data.shape)
    scaler = MinMaxScaler(feature_range=(0,1))
## This was scalling the data between 0-1
#    for i in range(4478):
#        for j in range(97):
#            data[i][j] = scaler.fit_transform((data[i][j]).reshape(-1,1))

    return data,labels

if __name__== '__main__':
    data,labels = GetForexData()
    print('Labels Shape is:',labels[0].size,'Data Shape is:',data[0].shape)
