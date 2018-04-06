import numpy as np

class IMSE (object):
    def __init__(self, ndims, nclass, _lambda):
        self.ndims = ndims
        self.nclass = nclass
        self.nc = np.ones(self.nclass)
        self._lambda = _lambda
        self.V = 0
        self.WW = 0
        self.n = 0
    
    def init_build(self, init_data):
        dataX = init_data[:,:-1]
        dataY = init_data[:,-1]
        ncount = np.size(dataX,0)
        self.n += ncount
        ndims = np.size(dataX,1)
        if not self.ndims == ndims:
            print("입력한 데이터의 차원이 맞지 않습니다")
            return 0
        class_check = np.array([])
        for i in range(ncount):
            class_check = np.append(class_check, dataY[i])
        exist_class = np.unique(class_check)
        if not len(exist_class) == self.nclass:
            print("class의 개수가 맞지 않습니다.")
            print("초기 데이터의 클래스 수 : {}, 초기데이터의 클래스 : {}".format(len(exist_class), exist_class)) 
            return 0
        else:
            y = np.eye(self.nclass)*self.n
            A = np.append(np.ones([self.nclass,1]), dataX, axis=1)
            self.M = np.linalg.inv(np.matmul(A.T, A)+self._lambda*np.eye(self.ndims+1))
            self.W = np.matmul(np.matmul(self.M,A.T),y)
        return np.matmul(A,self.W)

    def update_model(self, sample):
        dataX = sample[:self.ndims]
        dataY = sample[-1]
        self.n += 1
        temp = dataX.T
        temp_class = int(dataY)
        self.nc[temp_class] += 1
        y = np.zeros(self.nclass)
        y[temp_class] = self.n/np.sqrt(self.nc[temp_class])
        hat_x = np.expand_dims(np.append(np.array([1]), temp).T,axis=1)
        self.M = self.M - np.matmul(np.matmul((np.matmul(self.M,hat_x)/(1.+ np.matmul(np.matmul(hat_x.T,self.M),hat_x))),hat_x.T),self.M)
        self.W = self.W+np.matmul(np.matmul(self.M, hat_x),(y-np.matmul(hat_x.T,self.W)))
        return np.matmul(hat_x.T,self.W)
    
    def get_proj_data(self, testset):
        tmp = np.expand_dims(np.ones(np.size(testset,0)),axis=1)
        tmp = np.append(tmp, testset[:,:-1], axis=1)
        return np.matmul(tmp,self.W)






















    
    
