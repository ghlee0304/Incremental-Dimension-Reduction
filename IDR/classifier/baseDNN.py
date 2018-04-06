import tensorflow as tf

class DNN(object):
    def __init__(self, sess, ndims, nclass, layers, model_name):
        self.sess = sess
        self.model_name = model_name
        self.ndims = ndims
        self.nclass = nclass
        self.layers = layers
        self.attempt = 1
        self.h = []
        self.weight = []
        self.bias = []
        self._model_()
        
    def _model_(self):
        tf.set_random_seed(0)
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.float32, [None, self.ndims])
            self.Y = tf.placeholder(tf.int32, [None, 1])
            self.learning_rate = tf.placeholder(tf.float32)
            self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, self.nclass), [-1, self.nclass])

            nlayers = len(self.layers)

            if nlayers == 1:
                self.WF = tf.get_variable("W1", shape = [self.ndims, self.nclass], initializer = tf.glorot_uniform_initializer(seed=0))
                self.bF = tf.get_variable("b1", shape = [self.nclass], initializer =  tf.glorot_uniform_initializer(seed=0))
                self.a = tf.add(tf.matmul(self.X, self.WF),self.bF)
                self.h.append(self.a)
            else:
                self.WF = tf.get_variable("W1", shape = [self.ndims, self.layers[0]], initializer = tf.glorot_uniform_initializer(seed=0))
                self.bF = tf.get_variable("b1", shape = [self.layers[0]], initializer =  tf.glorot_uniform_initializer(seed=0))
                self.a = tf.matmul(self.X, self.WF)+self.bF
                self.h.append(tf.nn.relu(self.a))
            
            for i in range(1,nlayers):
                if i == nlayers-1:
                    self.weight.append(tf.get_variable("W{}".format(i+1), shape = [self.layers[i-1], self.nclass], initializer = tf.glorot_uniform_initializer(seed=0)))
                    self.bias.append(tf.get_variable("b{}".format(i+1), shape = [self.nclass], initializer =  tf.glorot_uniform_initializer(seed=0)))
                    a  = tf.matmul(self.h[-1], self.weight[-1])+self.bias[-1]
                    self.h.append(a)
                else:
                    self.weight.append(tf.get_variable("W{}".format(i+1), shape = [self.layers[i-1], self.layers[i]], initializer = tf.glorot_uniform_initializer(seed=0)))
                    self.bias.append(tf.get_variable("b{}".format(i+1), shape = [self.layers[i]], initializer = tf.glorot_uniform_initializer(seed=0)))
                    a  = tf.matmul(self.h[-1], self.weight[-1])+self.bias[-1]
                    self.h.append(tf.nn.relu(a))
            self.logits = self.h[-1]
            self.hypothesis = tf.nn.softmax(self.logits, name='hypothesis')
            self.cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y_one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.predict = tf.argmax(self.hypothesis, 1)
            correct_prediction = tf.equal(self.predict, tf.argmax(self.Y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.attempt +=1
            
    def train(self, x_data, y_data, learning_rate):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate : learning_rate})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def get_logits(self, x_test):
        return self.sess.run(self.WF, feed_dict={self.X: x_test})

    def get_pred(self,x_test):
        return self.sess.run(self.predict, feed_dict={self.X:x_test})

