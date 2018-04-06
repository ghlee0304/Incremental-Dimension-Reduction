#-*- coding: utf-8 -*-
from __future__ import division
from model.ccipca import *
from model.imse import *
from classifier.baseDNN import *
from utils import *
from AppController import *
import numpy as np
import matplotlib.pyplot as plt

train_set, test_set, nclass = data_load("digits", seed=0)

input_dim = np.size(train_set,1)-1
target_dim = 10
learning_rate = 1e-3

#CCIPCA
#initialize
sess = tf.Session()
appController = AppController()

m = CCIPCA(input_dim,target_dim)
clf = DNN(sess, target_dim, nclass, [target_dim,30,20], 'base1')
appController.train('ccipca', m, clf, sess, target_dim, train_set, test_set,learning_rate)
acc1 = appController.getAcc()

#no reduction
clf2 = DNN(sess,input_dim, nclass,[target_dim,30,20],'base2')
appController.train('default', 0, clf2, sess, target_dim, train_set, test_set,learning_rate)
acc2 = appController.getAcc()

m2 = IMSE(input_dim, nclass, 0.1)
clf3 = DNN(sess,  nclass, nclass,[nclass,30,20],'base3')
appController.train('imse', m2, clf3, sess, nclass, train_set, test_set,learning_rate)
acc3 = appController.getAcc()

plt.plot(acc2,c='b',label = "no reduction")
plt.plot(acc1,c='r',label = "CCIPCA")
plt.plot(acc3,c='g',label = "IMSE")
plt.legend()
plt.grid()
plt.show()
