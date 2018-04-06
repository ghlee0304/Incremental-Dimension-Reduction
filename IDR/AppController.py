import tensorflow as tf
import numpy as np 

class AppController(object):
    def __init__(self):
        self.acc=0
        self.m = 0
        self.clf = 0
        self.learning_rate = 0
    
    def train(self, algo_name, m, clf, sess, target_dim, train_set, test_set, learning_rate):
        self.m = m
        self.clf = clf
        self.learning_rate = learning_rate
        if not self.m==0:
            init_proj_data = m.init_build(train_set[:target_dim,:])
        else:
            init_proj_data = train_set[:target_dim,:-1]
            
        input_dim = np.size(train_set,1)-1
        sess.run(tf.global_variables_initializer())
        c,_ = self.clf.train(init_proj_data, np.expand_dims(train_set[:target_dim,-1],axis=1), self.learning_rate)

        a = self.getAcc2(test_set)
        print("\n<<< 차원 축소 알고리즘 {} 진행 >>>\n".format(algo_name)) 
        print("초기 데이터에대한 분류기 정확도 {:.2%}".format(a))
        print("초기 데이터에대한 분류기 비용 {:.6f}".format(c))
        
        acc = []
        for i in range(target_dim, np.size(train_set,0)):
            tmp_label = np.array([[int(train_set[i,-1])]])
            if not self.m==0:
                proj_data = self.m.update_model(train_set[i,:])
                #proj_data = np.expand_dims(proj_data,axis=0)
            else:
                proj_data = np.expand_dims(train_set[i,:-1],axis=0)

            c,_= self.clf.train(proj_data,tmp_label, self.learning_rate)
            a = self.getAcc2(test_set)
            acc.append(a)
            if i%100 == 0:
                print("Epoch {:6d}\t {:.2%}".format(i, a))
                print("Epoch {:6d}\t {:.6f}".format(i, c))
                
        a = self.getAcc2(test_set)

        print("최종 정확도 {:.2%}".format(a))
        self.acc = acc
        return acc

    def getAcc(self):
        return self.acc

    def getAcc2(self, test_set):
        if not self.m==0:
            a = self.clf.get_accuracy(self.m.get_proj_data(test_set), np.expand_dims(test_set[:,-1], axis=1))
        else:
            a = self.clf.get_accuracy(test_set[:,:-1], np.expand_dims(np.array(test_set[:,-1]), axis=1))
        return a
