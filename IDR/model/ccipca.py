import numpy as np

class CCIPCA(object):
    def __init__(self, ndims, K):
        self.ndims = ndims
        self.avg_x = np.zeros(self.ndims)
        self.K = K
        self.V = 0
        self.U = 0
        self.n = 1
    
    def init_build(self, init_data):
        dataX = init_data[:,:-1]
        dataY = init_data[:,-1]
        ncount = np.size(dataX,0)
        self.n += ncount-1
        ndims = np.size(dataX,1)
        if not self.ndims == ndims:
            print("입력한 데이터의 차원이 맞지 않습니다")
            return 0
        
        U = np.zeros([self.ndims, self.K]) #dxK
        U[:,0] = dataX[0,:].T
        self.avgX = dataX[0,:] # 1xd
        for i in range(1,self.K):
            temp = dataX[i,:]
            self.n += i
            self.avg_x = (self.n-1)/self.n*self.avg_x+1/self.n*temp
            x = temp - self.avg_x
            for j in range(0,min(i+1,self.K)):
                if (j == i):
                    U[:,j] = x.T
                else:
                    U[:,j] = ((self.n-1)/self.n)*U[:,j]+1/self.n*(x.T*(x*U[:,j]))/np.linalg.norm(U[:,j])
                    x = x-(x*U[:,j])*U[:,j].T/np.square(np.linalg.norm(U[:,j]))
        for i in range(0,self.K):
            U[:,i] = U[:,i]/np.linalg.norm(U[:,i])
        self.U = U;
        return np.matmul((dataX-self.avg_x),self.U)

    def update_model(self, sample):
        dataX = sample[:self.ndims]
        dataY = sample[-1]
        l = 2
        self.n += 1
        self.avg_x = (self.n-1)/self.n*self.avg_x+1/self.n*dataX
        x = dataX-self.avg_x
        for j in range(0,self.K):
            if (j  == self.n):
                self.U[:,j] = x.T
            else:
                self.U[:,j] = ((self.n-1)/self.n)*self.U[:,j]+1/self.n*(x.T*(x*self.U[:,j]))/np.linalg.norm(self.U[:,j])
                x = x-(x*self.U[:,j])*self.U[:,j].T/np.square(np.linalg.norm(self.U[:,j]))
        for i in range(0,self.K):
            self.U[:,i] = self.U[:,i]/np.linalg.norm(self.U[:,i])    
        return np.expand_dims(np.matmul(dataX-self.avg_x,self.U), axis=0)
    
    def get_proj_data(self, testset):
        return np.matmul(testset[:,:-1]-self.avg_x,self.U)
