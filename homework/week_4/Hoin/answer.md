1. Mandelbrot
<pre> <code>
%matplotlib inline
!pip install tensorboardcolab

import tensorboardcolab
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tbc = tensorboardcolab.TensorBoardColab(graph_path='./tensorboard')

tf.reset_default_graph()

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
init_value = X+1j*Y

xs = tf.constant(init_value, name='xs')
zs = tf.Variable(xs, name='zs')
zs_square = tf.multiply(zs, zs, name='zs_squre')
zs_add = tf.add(xs, zs_square, name="zs_add")
zs_ = tf.assign(zs, zs_add)

zs_abs = tf.abs(zs_add, name='zs_abs')
zs_less = tf.less(zs_abs, 4, name='zs_less')
zs_cast = tf.cast(zs_less, tf.float32)

xs_zeros = tf.zeros_like(xs, name='xs_zeros', dtype=tf.float32)
ns = tf.Variable(xs_zeros, name='ns')
ns_ = tf.assign_add(ns, zs_cast)
step = tf.group(zs_, ns_)

writer = tf.summary.FileWriter(logdir='./tensorboard')
writer.add_graph(graph=tf.get_default_graph())

tb_ns_mean = tf.summary.scalar(name='ns_mean', tensor=tf.reduce_mean(ns))
tb_ns_histogram = tf.summary.histogram(name='ns_histogram', values=ns)
tb_ns_image = tf.summary.image(name='ns_image', tensor=tf.reshape(ns, shape=[1, 520, 600, 1]))

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    merged_, _ = sess.run([merged, step])
    writer.add_summary(merged_, i)

writer.flush()

#�׷��� ����
saver = tf.train.Saver()
saver.save(sess, './tmp/mandelbrot')

#�׷��� �� ���� ����
tf.reset_default_graph()
tf.train.import_meta_graph('tmp/mandelbrot.meta')
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './tmp/mandelbrot')
graph = tf.get_default_graph()
zs = graph.get_tensor_by_name('zs:0')
ns = graph.get_tensor_by_name('ns:0')
</code></pre>


2.Julia
<pre> <code>
tbc = tensorboardcolab.TensorBoardColab(graph_path='./tensorboard')

tf.reset_default_graph()

Y,X = np.mgrid[-2:2:0.005, -2:2:0.005]
init_value = X + 1j * Y

xs = tf.constant(init_value, name='xs')
zs = tf.Variable(xs, name='zs')
zs_square = tf.multiply(zs, zs, name='zs_square')
zs_sub = tf.subtract(zs_square, 0+0.75j, name='zs_sub')
zs_ = tf.assign(zs, zs_sub)
zs_abs = tf.abs(zs_sub, name='zs_abs')
zs_less = tf.math.less(zs_abs, 4, name='zs_less')
zs_cast = tf.cast(zs_less, tf.float32)

xs_zeros = tf.zeros_like(xs, name='xs_zeros', dtype=tf.float32)
ns = tf.Variable(xs_zeros, name='ns')

ns_ = tf.assign_add(ns, zs_cast)

writer = tf.summary.FileWriter(logdir='./tensorboard')
writer.add_graph(graph=tf.get_default_graph())

tb_ns_mean = tf.summary.scalar(name='ns_mean', tensor=tf.reduce_mean(ns))
tb_ns_histogram = tf.summary.histogram(name='ns_histogram', values=ns)
tb_ns_image = tf.summary.image(name='ns_image', tensor=tf.reshape(ns, shape=[1, 800, 800, 1]))

merged = tf.summary.merge_all()

step = tf.group(zs_, ns_)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    merged_, _ = sess.run([merged, step])
    writer.add_summary(merged_, i)
result = sess.run(ns)
writer.flush()

plt.imshow(result)
plt.show()

#�׷��� ����
saver = tf.train.Saver()
saver.save(sess, './tmp/julia')

#�׷��� ����
tf.reset_default_graph()
tf.train.import_meta_graph('tmp/julia.meta')
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './tmp/julia')
graph = tf.get_default_graph()
zs = graph.get_tensor_by_name('zs:0')
ns = graph.get_tensor_by_name('ns:0')
</code> </pre>

3.rain drop
<pre> <code>
graph = tf.Graph()
with graph.as_default():
    eps = tf.placeholder(tf.float32, shape=())
    damping = tf.placeholder(tf.float32, shape=())

    # Create variables for simulation state
    U = tf.Variable(u_init, name='U')
    Ut = tf.Variable(ut_init, name='Ut')
    
    ### ������ �Ʒ��� �ۼ��� �ּ���
    laplace_U = laplace(U)
    
    U_ = U + eps * Ut
    Ut_ = (laplace(U) - Ut * damping) * eps + Ut

    # Operation to update the state
    update_op = tf.group(tf.assign(U, U_), tf.assign(Ut, Ut_))
    
    U_ = (U + tf.constant(0.1)) / tf.cast(tf.constant(0.2), tf.float32) * tf.constant(255.)
    output = tf.cast(tf.clip_by_value(U_, 0, 255), np.uint8)

</code> </pre>

4. KNN
<pre> <code>
graph = tf.Graph()
with graph.as_default():
    inX_tf = tf.placeholder(tf.float32, shape=(2), name='inX')
    dataset_tf = tf.placeholder(tf.float32, shape=(None,2), name='dataset')
    labels_tf = tf.placeholder(tf.string, shape=(None,), name='labels')
    K_tf = tf.placeholder_with_default(4,(),name='K')
    
    sub_ = tf.subtract(inX_tf, dataset_tf)
    add_ = tf.math.reduce_sum(tf.multiply(sub_, sub_), axis=1)
    dists = tf.math.sqrt(add_)
    
    sorted_index = tf.argsort(dists)
    K_labels = tf.boolean_mask(labels_tf, sorted_index < K_tf)
    
    unique_labels, _, count = tf.unique_with_counts(K_labels)
    max_count = tf.math.argmax(count)
    output = tf.gather_nd(unique_labels, [max_count])
</code></pre>

Kmeans
<pre><code>
k = 4

# (1) �߽��� �ʱ�ȭ
min_ = tf.math.reduce_min(dataset, axis=[0])
max_ = tf.math.reduce_max(dataset, axis=[0])
center = tf.random.uniform(shape=(k, 2), minval=min_, maxval=max_, dtype=tf.float64)
    
centroid = tf.Variable(center, name='centroid')

cluster_per_point = tf.Variable(tf.zeros(shape=dataset.shape[0], dtype=tf.int64), name='cluster_point')

# (2) �Ÿ� ��� 
diff_mat = tf.subtract(tf.reshape(centroid, (-1,1,2)), tf.reshape(dataset, (1,-1,2)))
dists = tf.sqrt(tf.reduce_sum(tf.multiply(diff_mat, diff_mat), axis=2))

# (3) �� �����͸� �Ÿ��� ���� ����� �������� �Ҵ�
cluster_ = tf.assign(cluster_per_point, tf.argmin(dists, axis=0))

# (4) �� ���� �� ������ ����� ��� ��, ������ �߽����� �ٽ� ���
# for in ��������
means_ = tf.zeros(shape=(1, 2), dtype=tf.float64)
for i in range(k):
    idx_ = tf.where(tf.equal(cluster_per_point, i))
    data_ = tf.gather(dataset, idx_)
    mean_ = tf.reshape(tf.reduce_mean(data_, axis=0), [1, 2])
    means_ = tf.concat([means_, mean_], axis=0)
means_ = tf.slice(means_, [1,0], [4,2])

update_centroides = tf.assign(centroid, means_)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    _, __, centroid_, cluster_point_ =sess.run([update_centroides, cluster_, centroid, cluster_per_point])
    
print(centroid_)

plt.scatter(dataset[:,0], dataset[:,1])
plt.scatter(centroid_[:,0], centroid_[:,1], marker='+')
plt.show()

</code></pre>

5. section 6
<pre><code>
# �켱 �������ּ���
graph = tf.Graph()
with graph.as_default():
    xs = tf.placeholder(tf.float32, shape=(None,), name='x')
    y_true = tf.placeholder(tf.float32, shape=(None,), name='y_true')

    # ���� �ʱ�ȭ
    with tf.variable_scope('weights'):
        W = tf.Variable(tf.random.normal([1]),"W")
        b = tf.Variable(tf.zeros([1]),'b')
        

    # Model
    with tf.variable_scope("Linear_Regression"):
        y_pred = W * xs + b
    y_pred = tf.identity(y_pred, name="y_pred")
    
    # Loss Function
    with tf.variable_scope("losses"):
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
    
    # Optimizer
    train_op = (tf.train
                .GradientDescentOptimizer(0.01)
                .minimize(loss))
    
log_dir = "./log/"
tbc = tensorboardcolab.TensorBoardColab(graph_path=log_dir)
with graph.as_default():
    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(logdir=log_dir)
    writer.add_graph(graph)
    
    tb_w = tf.summary.scalar(name='w_', tensor=tf.reduce_mean(W))
    tb_b = tf.summary.scalar(name='b_', tensor=tf.reduce_mean(b))
    tb_loss = tf.summary.scalar(name='tb_loss', tensor=loss)
    
    merged_all = tf.summary.merge_all()
    
    for step in range(1000):
        _, summary_values = sess.run([train_op, merged_all],
                                     feed_dict={
                                         xs:data_X,
                                         y_true:data_Y
                                     })
        writer.add_summary(summary_values, step)
    writer.flush()
</code></pre>