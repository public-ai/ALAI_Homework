### OVA
~~~
%matplotlib inline

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
~~~
~~~
iris = load_iris()
xs = iris['data']
ys = iris['target']
ys_name = iris['target_names']
~~~
~~~
class IrisIdentifer:
    def __init__(self, sess, target_name, target_index):
        self.sess = sess
        self.target_name = target_name
        self.target_index = target_index
        
        self.xs = iris['data']
        self.ys = iris['target']
        
        # xs normalization
        self.xs = (self.xs - np.min(self.xs, axis = 0)) / (np.max(self.xs, axis = 0) - np.min(self.xs, axis = 0)) # shape = (150,4)
        
        # ys 수정, 정답: 1, 오답: 0
        target_indices = np.where(ys == self.target_index)
        self.ys[target_indices] = 1
        other_indices = np.where(ys != self.target_index)
        self.ys[other_indices] = 0
        self.ys = tf.reshape(ys, [-1, 1])
        self.ys = tf.cast(ys, tf.float64)
        
        # 필요한 변수 선언.
        self.lr = 0.01
        self.acc = tf.Variable(-10, dtype = tf.float64)
        
        # layer1 (hidden)
        self.weights_1 = tf.Variable(tf.random.normal(shape = [4,10], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_1") # shape = (150, unit)
        self.bias_1 = tf.Variable(tf.zeros(shape = [10,], dtype = tf.float64), name = "bias_1")

        # layer2 (출력층)
        self.weights_2 = tf.Variable(tf.random.normal(shape = [10,1], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_2") # shape = (150, unit)
        self.bias_2 = tf.Variable(tf.zeros(shape = [1,], dtype = tf.float64), name = "bias_2")
        
        self.sess.run(tf.global_variables_initializer())
        
        
    def train(self):
        z_1 = tf.matmul(self.xs, self.weights_1) + self.bias_1
        a_1 = tf.nn.relu(z_1)
            
        z_2 = tf.matmul(a_1, self.weights_2) + self.bias_2
        a_2 = tf.nn.sigmoid(z_2)

        a_2_ = tf.round(a_2) # a_2  0, 1 로 변환
        
        update_acc = tf.assign(ref = self.acc, value = tf.math.reduce_mean(tf.cast(tf.equal(a_2_, self.ys), tf.float64)))
  
        self.ys = tf.reshape(self.ys, [-1, 1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= a_2, labels= self.ys)

        optimize = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
    
        self.sess.run([optimize, update_acc])
~~~
~~~
sess = tf.Session()
irisIdentifier_1 = IrisIdentifer(sess, "Setosa", 0)
irisIdentifier_2 = IrisIdentifer(sess, "Versicolour", 1)
irisIdentifier_3 = IrisIdentifer(sess, "Virginica", 2)

for i in np.arange(100):
    irisIdentifier_1.train()
    irisIdentifier_2.train()
    irisIdentifier_3.train()
~~~

### week5_7_Logistic_Regression.ipynb
~~~
# init
# min-max normalization
xs = iris['data']
ys = iris['target']
ys_name = iris['target_names']

xs1 = xs[ys == 0]
xs2 = xs[ys == 1]
xs = np.concatenate([xs1, xs2], axis = 0)
xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))

ys1 = ys[ys == 0]
ys2 = ys[ys == 1]
ys = np.concatenate([ys1, ys2], axis = 0)
ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))

# weight 초기화
w_0, w_1, w_2, w_3, w_4 = np.random.random(5)
# 학습률
alpha = 0.1


w_0_history = [w_0]
w_1_history = [w_1]
w_2_history = [w_2]
w_3_history = [w_3]
w_4_history = [w_4]

loss_history = []
acc_history = []

for i in np.arange(100):
    old_w_0 = w_0_history[-1]
    old_w_1 = w_1_history[-1]
    old_w_2 = w_2_history[-1]
    old_w_3 = w_3_history[-1]
    old_w_4 = w_4_history[-1]

    z = old_w_0 + old_w_1 * xs[:,0] + old_w_2 * xs [:,1] + old_w_3 * xs[:,2] + old_w_4 * xs[:,3]
    prob = 1/(1+np.exp(-z))

    dw0 = np.mean(prob - ys)
    dw1 = np.mean((prob - ys) * xs[:,0])
    dw2 = np.mean((prob - ys) * xs[:,1])
    dw3 = np.mean((prob - ys) * xs[:,2])
    dw4 = np.mean((prob - ys) * xs[:,3])
    new_w_0 = old_w_0 - alpha * dw0
    new_w_1 = old_w_1 - alpha * dw1
    new_w_2 = old_w_2 - alpha * dw2
    new_w_3 = old_w_3 - alpha * dw3
    new_w_4 = old_w_4 - alpha * dw4

    crossentorpy = - np.mean(ys * np.log(prob) + (1 - ys)*np.log(1 - prob))
    loss_history.append(crossentorpy)

    acc_count = 0
    for i in np.arange(100):
        if (prob[i] >= 0) & (prob[i] <= 0.5) & (ys[i] >= 0) & (ys[i] <= 0.5):
            acc_count += 1
        elif (prob[i] > 0.5) & (prob[i] <= 1) & (ys[i] > 0.5) & (ys[i] <= 1):
            acc_count += 1
    acc_history.append(acc_count/100)
    
    w_0_history.append(new_w_0)
    w_1_history.append(new_w_1)
    w_2_history.append(new_w_2)
    w_3_history.append(new_w_3)
    w_4_history.append(new_w_4)
   

plt.subplot(121)
plt.plot(np.arange(100), loss_history)
plt.subplot(122)
plt.plot(np.arange(100), acc_history)
plt.show
~~~

### week5_8_Model_Evaluation
~~~
cut_values = np.linspace(0.0, 1.0, 100)

TPR_history = []
FPR_history = []
result_cut_value = -1
min_FPR = -1

for cut_value in cut_values:
    TPR = np.sum((df.iloc[:,0] >= cut_value )& (df.iloc[:,1] == 0))/ (np.sum(df.iloc[:,1] == 0))
    FPR = np.sum((df.iloc[:,0] >= cut_value )& (df.iloc[:,1] == 1))/ (np.sum(df.iloc[:,1] == 1))
    TPR_history.append(TPR)
    FPR_history.append(FPR)

plt.plot(TPR_history, FPR_history)
plt.show()
~~~


### week6_3_Tensorflow을_이용한_FeedForward_Network_구현하기
~~~
# [ train ]

# xs:  (506, 13)
# init
weights_1 = tf.Variable(tf.random.normal(shape = [13, 64], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_1")
bias_1 = tf.Variable(tf.zeros(shape = [64], dtype=tf.dtypes.float64), name = "bias_1")

weights_2 = tf.Variable(tf.random.normal(shape = [64, 128], mean = 0.0, stddev = 0.1,  dtype = tf.float64), name = "weights_2")
bias_2 = tf.Variable(tf.zeros(shape = [128], dtype=tf.dtypes.float64), name = "bias_2")

weights_3 = tf.Variable(tf.random.normal(shape = [128, 256], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_3")
bias_3 = tf.Variable(tf.zeros(shape = [256], dtype=tf.dtypes.float64), name = "bias_3")

weights_4 = tf.Variable(tf.random.normal(shape = [256, 1], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_3")
bias_4 = tf.Variable(tf.zeros(shape = [1], dtype=tf.dtypes.float64), name = "bias_3")


train_loss = tf.Variable(0, name = "loss", dtype = tf.dtypes.float64)

# hidden
# layer1
z_1 = tf.matmul(train_xs, weights_1) + bias_1
result_a_1 = tf.nn.relu(z_1)

# layer2
z_2 = tf.matmul(result_a_1, weights_2) + bias_2
result_a_2 = tf.nn.relu(z_2)

# layer3
z_3 = tf.matmul(result_a_2, weights_3) + bias_3
result_a_3 = tf.nn.relu(z_3)


# output
z_4 = tf.matmul(result_a_3, weights_4) + bias_4
hat_ys = tf.math.sigmoid(-z_4)
#hat_ys = 1./(1 + tf.log(-z_4)) -- 왜 nan?
loss_assign = tf.assign(loss, tf.reduce_mean(tf.math.square(hat_ys - train_ys)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf.reduce_mean(tf.math.square(hat_ys - train_ys)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in np.arange(1000):
    sess.run([result_a_1, result_a_2, result_a_3, loss_assign, train])
    
# [ test ]
# init
weights_1 = tf.Variable(tf.random.normal(shape = [13, 64], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_1")
bias_1 = tf.Variable(tf.zeros(shape = [64], dtype=tf.dtypes.float64), name = "bias_1")

weights_2 = tf.Variable(tf.random.normal(shape = [64, 128], mean = 0.0, stddev = 0.1,  dtype = tf.float64), name = "weights_2")
bias_2 = tf.Variable(tf.zeros(shape = [128], dtype=tf.dtypes.float64), name = "bias_2")

weights_3 = tf.Variable(tf.random.normal(shape = [128, 256], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_3")
bias_3 = tf.Variable(tf.zeros(shape = [256], dtype=tf.dtypes.float64), name = "bias_3")
print()
weights_4 = tf.Variable(tf.random.normal(shape = [256, 1], mean = 0.0, stddev = 0.1, dtype = tf.float64), name = "weights_3")
bias_4 = tf.Variable(tf.zeros(shape = [1], dtype=tf.dtypes.float64), name = "bias_3")


test_loss = tf.Variable(0, name = "loss", dtype = tf.dtypes.float64)

# hidden
# layer1
z_1 = tf.matmul(test_xs, weights_1) + bias_1
result_a_1 = tf.nn.relu(z_1)

# layer2
z_2 = tf.matmul(result_a_1, weights_2) + bias_2
result_a_2 = tf.nn.relu(z_2)

# layer3
z_3 = tf.matmul(result_a_2, weights_3) + bias_3
result_a_3 = tf.nn.relu(z_3)


# output
z_4 = tf.matmul(result_a_3, weights_4) + bias_4
hat_ys = tf.math.sigmoid(-z_4)
loss_assign = tf.assign(loss, tf.reduce_mean(tf.math.square(hat_ys - test_ys)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf.reduce_mean(tf.math.square(hat_ys - test_ys)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in np.arange(1000):
    sess.run([result_a_1, result_a_2, result_a_3, loss_assign, train])
    print(sess.run(loss))
~~~
