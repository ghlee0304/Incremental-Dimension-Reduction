import numpy as np 
from sklearn import datasets

def data_load(data_name, seed):
    if data_name == "covtype":
        temp_dataset = datasets.fetch_covtype()
        dataX = temp_dataset.data
        dataY = temp_dataset.target-1
    elif data_name == "digits":
        temp_dataset = datasets.load_digits()
        dataX = temp_dataset.data
        dataY = temp_dataset.target
        
    nclass = len(np.unique(dataY))
    np.random.seed(seed)
    mask = np.random.permutation(np.size(dataX,0))
    dataX = dataX[mask]
    dataY = dataY[mask]
    ntrain = int(np.size(dataX,0)*0.8)
    
    testX = dataX[ntrain:]
    testY = dataY[ntrain:]
    dataX = dataX[:ntrain]
    dataY = dataY[:ntrain]
    dataY = np.expand_dims(dataY,axis=1)
    testY = np.expand_dims(testY,axis=1)

    ndims = np.size(dataX,1)
    nclass = len(np.unique(dataY))

    _count = 0
    _idx = []
    for i in range(len(dataY)):
        if _count == dataY[i]:
            tempx = np.expand_dims(dataX[i,:],axis=0)
            tempy = np.expand_dims(dataY[i],axis=0)
            _idx.append(i)
            if _count == 0:
                temp_datax = tempx
                temp_datay = tempy
                _count += 1
            else:
                temp_datax = np.append(temp_datax, tempx, axis=0)
                temp_datay = np.append(temp_datay, tempy, axis=0)
                _count += 1
        if _count == nclass:
            break
    _idx = np.array(_idx)

    trainX = np.delete(dataX,_idx,0)
    trainY = np.delete(dataY,_idx,0)

    dataX = 0 #free(dataX)
    dataY = 0
    
    trainX = np.append(temp_datax,trainX,axis=0)
    trainY = np.append(temp_datay,trainY,axis=0)

    temp_datax = 0
    temp_datay = 0

    train_set = np.append(trainX, trainY, axis=1)
    test_set = np.append(testX, testY, axis=1)
    return train_set, test_set, nclass

