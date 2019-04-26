### 1. 망델브롯 Fractal 구현하고 텐서 보드에 연결하기
~~~ Python
tf.reset_default_graph()


tbc=tensorboardcolab.TensorBoardColab(graph_path='./tensorboard')
writer = tf.summary.FileWriter(logdir='./tensorboard')



Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
init_value = X + 1j*Y

xs = tf.constant(init_value, name = "xs")
xs_zeros = tf.zeros_like(xs, tf.float32)
ns = tf.Variable(xs_zeros, name = "ns")

zs = tf.Variable(xs)
zs_square = tf.multiply(x = zs, y = zs, name = "zs_square")
zs_add = tf.add(x = xs, y = zs_square, name = "zs_add")
zs = tf.assign(zs, zs_add)

zs_abs = tf.abs(zs_add, name = "zs_abs")
zs_loss = tf.less(x = zs_abs, y = 4, name = "zs_less")
zs_cast = tf.cast(zs_loss, dtype = tf.float32, name = "zs_cast")
add = tf.assign_add(ns, zs_cast)
group = tf.group(add, zs)


ns_mean = tf.reduce_mean(ns)
tf.summary.scalar("ns_mean", ns_mean)
tf.summary.histogram("ns_histogram", ns)
tf.summary.image("ns_image", tf.reshape(ns, shape = [1, 520, 600, 1]))

summaries = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(logdir='./tensorboard')
for i in range(100): # coach : 여러번 수행해야 mandelbrot을 그릴수 있습니다.
  # summ = sess.run(summaries) # coach : 실제 mandelbrot을 그리는 step을 빼먹으셨습니다.
  _, summ = sess.run([group, summaries]) 

  writer.add_graph(graph=tf.get_default_graph())
  writer.add_summary(summ, i)



# graph 저장
saver = tf.train.Saver()
saver.save(sess, './tmp/model1')

# graph 복원
tf.reset_default_graph()
tf.train.import_meta_graph('tmp/model1.meta')

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './tmp/model1')
graph = tf.get_default_graph()
~~~

### 2. Julia fractal 구현하기.
~~~
tf.reset_default_graph()


tbc=tensorboardcolab.TensorBoardColab(graph_path='./tensorboard')
writer = tf.summary.FileWriter(logdir='./tensorboard')

Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]
init_value = X + 1j*Y

xs = tf.constant(init_value, name = "xs")
zs = tf.Variable(xs, name = "zs")
zs_square = tf.multiply(zs, zs, name = "zs_square")
zs_sub = tf.subtract(zs_square, y = 0 + 0.75j, name = "zs_sub")
zs_assign = tf.assign(ref = zs, value = zs_sub)

zs_abs = tf.abs(zs_sub, name = "zs_abs")
zs_less = tf.math.less(x = zs_abs, y = 4, name = "zs_less")
zs_cast = tf.cast(zs_less, dtype = tf.float32, name = "zs_cast")


xs_zeros = tf.zeros_like(xs, name = "xs_zeors", dtype = tf.float32)
ns = tf.Variable(xs_zeros, name = "ns")
ns_assign = tf.assign_add(ref = ns, value = zs_cast)

group = tf.group(ns_assign, zs_assign)


ns_mean = tf.reduce_mean(ns)
tf.summary.scalar("ns_mean", ns_mean)
tf.summary.histogram("ns_histogram", ns)
tf.summary.image("ns_image", tf.reshape(ns, shape = [1, 800, 800, 1]))

summaries = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer.add_graph(graph=tf.get_default_graph())

for i in range(200): # coach : 1번 과제와 같은 문제입니다
  _, summ = sess.run([group, summaries])
  writer.add_summary(summ, i)

ns_result = sess.run(ns)
plt.imshow(ns_result)
plt.show()

# graph 저장
saver = tf.train.Saver()
saver.save(sess, './tmp/model2')

# graph 복원
tf.reset_default_graph()
tf.train.import_meta_graph('tmp/model2.meta')

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './tmp/model2')
graph = tf.get_default_graph()
~~~

### 3. rain Drop
~~~
# 학습 그래프
graph = tf.Graph()
with graph.as_default():
    eps = tf.placeholder(tf.float32, shape=())
    damping = tf.placeholder(tf.float32, shape=())

    # Create variables for simulation state
    
    # U = U + eps*Ut
    U = tf.Variable(u_init, name='U')
    Ut = tf.Variable(ut_init, name='Ut')
    
    value = tf.multiply(x = Ut, y = eps, name = "value")
    add_ = tf.add(x = U, y = value, name = "add_")
    result1 = tf.assign(U, add_)
    
    multi_2 = tf.multiply(x = Ut, y = damping, name = "multi_2")
    sub_2 = tf.subtract(x = laplace(U), y = multi_2, name = "sub_2")
    multi_2_2 = tf.multiply(x = eps, y = sub_2, name = "multi_2_2")
    add_2 = tf.add(x = multi_2_2, y = Ut, name = "add_2")
    result2 = tf.assign(Ut, add_2)
    
    
    update_op = tf.group(result1, result2, name = 'step')
    
    
    
    add = tf.math.add(x = U, y = 0.1)
    division = tf.math.divide(x = add, y = 0.2)
    multi = tf.math.multiply(x = division, y = 255)
    output = tf.clip_by_value(multi, clip_value_min = 0, clip_value_max = 255)
 ~~~
 
 ### 4. (1) Kmeans
 ~~~
 tf.reset_default_graph()

k = 4

dataset_tf = tf.Variable(dataset, name = "dataset_tf")

# 초기화
min_x = tf.reduce_min(dataset_tf[:, 0])
max_x = tf.reduce_max(dataset_tf[:, 0])

min_y = tf.reduce_min(dataset_tf[:, 1])
max_y = tf.reduce_max(dataset_tf[:, 1])

centroids = []
for i in range(0, k):
    centroid_x = tf.random.uniform(shape = [], minval = min_x, maxval = max_x, dtype = tf.float64)
    centroid_y = tf.random.uniform(shape = [], minval = min_y, maxval = max_y, dtype = tf.float64)
    centroids.append([centroid_x, centroid_y])
    
    
centroids_tf = tf.Variable(centroids, name = "centroids_tf")    

dists = tf.sqrt(tf.reduce_sum((tf.reshape(centroids_tf, [-1, 1, 2]) - tf.reshape(dataset_tf, [1, -1, 2]))**2, axis = -1))
cluster_per_point = tf.math.argmin(dists, axis = 0) # dataset 별로 가장 작은 i(k)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

colors = ['r', 'g', 'b', 'y']
for i in range(0, k):
    correct_dataset_tf = tf.boolean_mask(dataset_tf, tf.equal(cluster_per_point, i))
    centroids[i] = tf.math.reduce_mean(correct_dataset_tf, axis = 0)

update = tf.assign(centroids_tf, tf.convert_to_tensor(centroids))

for i in range(100):
    sess.run(update)
~~~

### 4-2. KNN
~~~
graph = tf.Graph()
with graph.as_default():
    inX_tf = tf.placeholder(tf.float32, shape=(2), name='inX')
    dataset_tf = tf.placeholder(tf.float32, shape=(None,2), name='dataset')
    labels_tf = tf.placeholder(tf.string, shape=(None,), name='labels')
    K_tf = tf.placeholder_with_default(4,(),name='K')

    distances = tf.math.reduce_sum((dataset_tf - inX_tf)**2, axis = -1)
    sorted_indices = tf.argsort(distances)
    k_indices = sorted_indices[:K_tf] # 가장 가까운 k개의 index 구함
    k_labels_tf = tf.gather(labels_tf, k_indices)
    
    k_lables, indx, count = tf.unique_with_counts(k_labels_tf, out_idx = tf.int32)
    
    output = k_lables[tf.argsort(count, direction='DESCENDING')[0]]
    result = tf.argsort(count)
~~~

### 5. Excercise6
~~~
# 정답을 입력해주세요
# 아래 학습 과정 중에 텐서보드를 선언하고
# Summary 연산을 통해 올려 주세요
with graph.as_default():
    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(logdir = log_dir)
    writer.add_graph(graph=tf.get_default_graph())
    
#     tb_w = tf.summary.scalar(name = 'W', tensor = W[0])
#     tb_b = tf.summary.scalar(name = 'b', tensor = b[0])
    
#     tb_loss = tf.summary.scalar(name = "loss", tensor = loss)
    
    tf.summary.scalar(name = 'W', tensor = W[0])
    tf.summary.scalar(name = 'b', tensor = b[0])
    
    tf.summary.scalar(name = "loss", tensor = loss)
    merged_all = tf.summary.merge_all()
   
    for step in range(1000):
        _, summary_values = sess.run([train_op, merged_all],
                                     feed_dict={
                                         xs:data_X,
                                         y_true:data_Y
                                     })
        writer.add_summary(summary_values, step)
   
    writer.flush()
~~~

