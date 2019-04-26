```python
Y , X = np.mgrid[-2:2:0.005, -2:2:0.005]
init_value = X +1j*Y

xs = tf.constant(init_value, name = 'xs')

zs = tf.Variable(xs, name = 'zs')
zs_square = tf.multiply(zs, zs, name= 'zs_square')
zs_sub = tf.subtract(zs_square, 0+0.75j, name = 'zs_sub')
zs_abs = tf.abs(zs_sub, name = 'zs_abs')
zs_less = tf.math.less(zs_abs, 4, name= 'zs_less')
zs_cast = tf.cast(zs_less, tf.float32)

xs_zeros = tf.zeros_like(xs, dtype = tf.float32, name = 'xs_zeros')
ns = tf.Variable(xs_zeros, name = 'ns')

step = tf.group(tf.assign(zs, zs_sub), tf.assign_add(ns, zs_cast))
```

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    sess.run(step)
result = sess.run(ns)
```

```python
# Julia fractal 시각화
plt.imshow(result)
plt.show()
```

```python
# Julia fractal을 텐서보드에 연결하기
# Julia fractal을 학습 시킨 후 그 과정을 Tensorboard와 연결하는 과정

tf.reset_default_graph()

Y , X = np.mgrid[-2:2:0.005, -2:2:0.005]
init_value = X +1j*Y

with tf.name_scope('cal'):
    xs = tf.constant(init_value, name = 'xs')
    zs = tf.Variable(xs, name = 'zs')
    xs_zeros = tf.zeros_like(xs, dtype = tf.float32, name = 'xs_zeros')
    ns = tf.Variable(xs_zeros, name = 'ns')
    
ns_image_tb = tf.summary.image(name = 'ns_image',
                              tensor = tf.reshape(ns, shape = [1, 800,800, 1]))
ns_mean_tb = tf.summary.scalar(name = 'ns_mean',
                              tensor = tf.reduce_mean(ns))
ns_hist_tb = tf.summary.histogram(name = 'ns_hist',
                                 values = ns)

zs_square = tf.multiply(zs, zs, name= 'zs_square')
zs_sub = tf.subtract(zs_square, 0+0.75j, name = 'zs_sub')
zs_abs = tf.abs(zs_sub, name = 'zs_abs')
zs_less = tf.math.less(zs_abs, 4, name= 'zs_less')
zs_cast = tf.cast(zs_less, tf.float32)

step = tf.group(tf.assign(zs, zs_sub), tf.assign_add(ns, zs_cast), name = 'step')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
tbs = tf.summary.merge_all()

tbc = tensorboardcolab.TensorBoardColab(graph_path = './julia')
writer = tf.summary.FileWriter(logdir = './julia')
writer.add_graph(tf.get_default_graph())

for i in range(200):
    _, tbs_ = sess.run([step, tbs])
    writer.add_summary(tbs_, global_step = i)
    
writer.flush()

saver.save(sess,
          save_path = './model/julia')
value = sess.run(ns)
plt.imshow(value)
plt.show()
```

```python
# julia fractal 복원하기
tf.reset_default_graph()
saver = tf.train.import_meta_graph('/model/julia.meta')
sess = tf.Session()
saver.restore(sess, '/model/julia')
ns = tf.get_default_graph().get_tensor_by_name('cal/ns:0')
step = tf.get_default_graph().get_operation_by_name('step')
sess.run('step')
ns_ = sess.run(ns)

plt.imshow(ns_)
plt.show
```

​	