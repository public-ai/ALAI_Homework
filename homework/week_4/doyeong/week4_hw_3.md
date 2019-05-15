```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

```python
def laplace(x):
    def make_kernel(a):
        """Transform a 2D array into a convolution kernel"""
        a = np.asarray(a)
        a = a.reshape(list(a.shape) + [1, 1])
        return tf.constant(a, dtype=1)

    def simple_conv(x, k):
        """A simplified 2D c1onvolution operation"""
        x = tf.expand_dims(tf.expand_dims(x, 0), -1)
        y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
        return y[0, :, :, 0]
    """Compute the 2D laplacian of an array"""

    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


# 임의의 좌표에 임의의 값 40개를 생성 합니다. 
N = 500
u_init = np.zeros([500,500] , dtype=np.float32)
ut_init = np.zeros([500,500] , dtype=np.float32)

for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()
```

```python
graph = tf.Graph()
with graph.as_default():
    eps = tf.placeholder(tf.float32, shape=())
    damping = tf.placeholder(tf.float32, shape=())

    # Create variables for simulation state
    U = tf.Variable(u_init, name='U')
    Ut = tf.Variable(ut_init, name='Ut')
    
    U_mul = tf.multiply(eps, Ut, name='U_mul')
    value_1 = tf.add(U, U_mul, name='value_1')
    tf.assign(U, value_1)
    Ut_mul_1 = tf.multiply(damping, Ut, name='Ut_mul_1')
    Ut_sub = tf.subtract(laplace(U), Ut_mul_1, name='Ut_sub')
    Ut_mul_2 = tf.multiply(eps, Ut_sub, name='Ut_mul_2')
    value_2 = tf.add(Ut, Ut_mul_2, name='value_2')
    tf.assign(Ut, value_2)
    update_op = tf.group( tf.assign(U, value_1) , tf.assign(Ut, value_2))
    
    # 평가 그래프 구현
    eva_1 = tf.add(U, 0.1, name='eva_1')
    eva_2 = tf.divide(eva_1, 0.2, name='eva_2')
    eva_3 = tf.multiply(eva_2, 255, name='eva_3')
    output = tf.clip_by_value(eva_3, 0, 255, name='output')
```

```python
with graph.as_default():
    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(update_op, {eps: 0.03, damping: 0.04})

    result = sess.run(output)
plt.imshow(result, cmap='Greys')
plt.show()
```

