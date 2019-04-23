1. 망델브롯

```python3

Y,X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
init_value = X + 1j*Y
#print(type(init_value))
graph2 = tf.Graph()
with graph2.as_default() :
    xs = tf.constant(init_value , name='xs')
    #tf.initialize_variables(xs)
    zs = tf.Variable(xs)
    #########################
    xs_zeros = tf.zeros_like(xs, name='xs_zeros', dtype = tf.float32)
    ns = tf.Variable(xs_zeros, name='ns')
    ns_mean = tf.reduce_mean(ns)
    print(tf.shape(ns))
    print(ns)
    ns_image = tf.reshape(ns, [-1, 520, 600, 1])
    
    ns_mean_ = tf.summary.scalar(name = 'ns_mean', tensor = ns_mean)
    ns_histogram_ = tf.summary.histogram(name = 'ns_histogram',values = ns)
    ns_image_ = tf.summary.image(name = 'ns_image', tensor =ns_image)
    #########################
    
    zs_square = tf.multiply(zs, zs, name= 'zs_square')
    zs_add = tf.add(xs, zs_square, name='zs_add')
    as_zs = tf.assign(ref=zs, value=zs_add)#
    zs_abs = tf.abs(zs_add, name='zs_abs')
    zs_less = tf.math.less(x = zs_abs, y = 4, name='zs_less')
    zs_cast = tf.cast(x=zs_less, dtype=tf.float32, name='zs_cast')
    as_ns = tf.assign_add(ref=ns, value=zs_cast)#
    
    ##########################################
    step = tf.group(as_zs, as_ns, name='step')
    #########################################
    
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(200) :
            merged_, step_ = sess.run([merged, step])
            writer.add_summary(merged_, i)
    
    writer.flush()

```

2. Julia fractal
```python3

Y,X = np.mgrid[-2:2:0.005, -2:2:0.005]
init_value = X + 1j*Y
graph3 = tf.Graph()
with graph3.as_default() :
    xs = tf.constant(init_value, name="xs")
    zs = tf.Variable(xs, name='zs')
    xs_zeros = tf.zeros_like(xs, name='xs_zeros', dtype=tf.float32)
    ns = tf.Variable(xs_zeros, name = 'ns')
    ns_mean = tf.reduce_mean(ns)
    print(ns)
    ns_ima = tf.reshape(ns, [-1,800,800,1])
    zs_square = tf.multiply(zs, zs, name='zs_square')
    zs_sub = tf.subtract(zs_square, 0+0.75j, name = 'zs_sub')
    zs_asig = tf.assign(ref=zs, value=zs_sub)
    zs_abs = tf.abs(zs_sub, name = 'zs_abs')
    zs_less = tf.math.less(x = zs_abs, y = 4, name='zs_less')
    zs_cast = tf.cast(x = zs_less, dtype=tf.float32, name='zs_cast')
    ns_asig2 = tf.assign_add(ref=ns, value= zs_cast)
    #zs_asig2 = tf.assign_add(ref=zs, value= zs_sub)

    tf.summary.scalar(name = 'ns_mean', tensor = ns_mean)
    tf.summary.image(name = 'ns_ima' , tensor = ns_ima)
    tf.summary.histogram(name = 'ns_hist', values = ns)

    step = tf.group(zs_asig, ns_asig2, name='step')

    merged = tf.summary.merge_all()
with tf.Session(graph = graph3) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        _, merged_ = sess.run([step, merged])
        writer.add_summary(merged_, i)
    writer.flush()

```

3. Rain drop
```python3
graph = tf.Graph()
with graph.as_default():
    eps = tf.placeholder(tf.float32, shape=())
    damping = tf.placeholder(tf.float32, shape=())

    # Create variables for simulation state
    U = tf.Variable(u_init, name='U', dtype=tf.float32)
    Ut = tf.Variable(ut_init, name='Ut', dtype=tf.float32)
    
    ### 정답을 아래에 작성해 주세요
    # U = U + eps*Ut
    mul_u = tf.multiply(eps, Ut)
    sum_u = tf.add(mul_u, U)
    assign_1 = tf.assign(ref=U, value=sum_u)
    
    mul_ut = tf.multiply(Ut, damping)
    sub_ut = tf.subtract(laplace(U), mul_ut) #부호이슈
    mul_ut2 = tf.multiply(sub_ut, eps)
    sum_ut = tf.add(Ut, mul_ut2)
    assign_2 = tf.assign(ref=Ut,value=sum_ut)
    update_op = tf.group(assign_1, assign_2, name = "step")
    
    sum_t = tf.add(U, 0.1)
    div_t = tf.divide(sum_t, 0.2)
    mul_t = tf.multiply(div_t, 255)
    output = tf.clip_by_value(mul_t, clip_value_min=0., clip_value_max=255.)
    
```

4-1. Kmeans
```python3
graph = tf.Graph()
centroids_ = []
with graph.as_default() :
    dataset_tf = tf.constant(dataset, dtype = tf.float32)
    
    min_xy = tf.reduce_min(dataset_tf, axis = 0)
    
    max_xy = tf.reduce_max(dataset_tf, axis = 0)
    
    ran_x = tf.random_uniform([4], minval = min_xy[0], maxval = max_xy[0])
    
    ran_y = tf.random_uniform([4], minval = min_xy[1], maxval = max_xy[1])
    
    centroids = tf.stack([ran_x, ran_y], axis = 1)
    
    with tf.Session() as sess:
        centroids_ = sess.run([centroids])
centroids_ = np.array(centroids_)
print('centroids_ :',centroids_.shape)


graph2 = tf.Graph()

with graph2.as_default() :
    
    cent1 = tf.Variable(centroids_, name='centroids', dtype=tf.float64)
    
    dataset_tf = tf.constant(dataset, shape=[80,2], dtype=tf.float64)
    
    dataset_tf = tf.reshape(dataset_tf, shape=[80,1,2])
    
    sub_tf = tf.subtract(dataset_tf, cent1)
    
    square_tf = tf.square(sub_tf)
    
    sum_tf = tf.reduce_sum(square_tf,axis = 2)
    
    dist_min_idx = tf.argmin(sum_tf, axis=1)
    
    dist_min_idx_f = tf.placeholder(shape=[80], dtype = tf.float64)
    
    k_ = tf.placeholder(shape=[], dtype = tf.float64)
    
    where_tf = tf.equal(dist_min_idx_f, k_)
    
    gather_tf = tf.boolean_mask(dataset_tf, where_tf)
    
    cen_tf = tf.reduce_mean(gather_tf, axis = 0)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000) :
            dist_min_idx_ = sess.run(dist_min_idx)
            cent_ = []
            for i2 in range(4) :
                cen_tf_ = sess.run(cen_tf,
                                   feed_dict={dist_min_idx_f : dist_min_idx_
                                                ,k_ : i2})
                cent_.append(cen_tf_)
                #print(i2,cen_tf_)
                
            cent_ = np.asarray(cent_)
            centroids_1=np.squeeze(centroids_)
            cent_1=np.squeeze(cent_)
            #print(centroids_1.shape) # 1 4 2
            #print(cent_1.shape)      # 4 1 2
            if (np.all(centroids_1 == cent_1))  :
                break
            else :
                centroids_ = cent_
                cent_con = tf.constant(cent_)
                cent_con = tf.transpose(cent_con, perm = [1,0,2])
                #print(sess.run(tf.shape(cent_con)))
                sess.run(tf.assign(ref=cent1, value=cent_con))

#print(centroids_)
#print(centroids_.shape)
centroids_=np.squeeze(centroids_)
#print(centroids_)
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.scatter(centroids_[:, 0], centroids_[:, 1], marker='+')
plt.show()

```

4-2. KNN
```python3
graph = tf.Graph()
with graph.as_default():
    inX_tf = tf.placeholder(tf.float32, shape=(2), name='inX')
    dataset_tf = tf.placeholder(tf.float32, shape=(None,2), name='dataset')
    labels_tf = tf.placeholder(tf.string, shape=(None,), name='labels')
    K_tf = tf.placeholder_with_default(4,(),name='K')
    
    ##########
    # CODE HERE!
    # 아래의 결과가 나오도록 수정해 주세요!
    ##########
    
    tf_sub = tf.subtract(inX_tf, dataset_tf, name="sub")
    tf_squ = tf.square(tf_sub, name="square")
    tf_sum = tf.reduce_sum(tf_squ, axis=1, name='sum')
    tf_sqrt = tf.sqrt(tf_sum, name="sqrt")
    
    #거리에 대해 인덱스 정렬 반환
    arg_sort = tf.argsort(tf_sqrt, axis = -1, name = 'arg_sorted')
    
    #인덱스 기반 labels_tf 서치
    #gath_tf = tf.gather_nd(params = labels_tf, indices = arg_sort, name='gath_tf')
    gath_tf = tf.gather(params = labels_tf, indices = arg_sort)
    
    la_sa = tf.rank(labels_tf)
    ar_sa = tf.rank(arg_sort)
    slice_tf = gath_tf[:K_tf] #tf.slice(gath_tf, [0], K_tf)   
    y, idx, count = tf.unique_with_counts(slice_tf)
    output = tf.gather(y, tf.argmax(count))
    
    with tf.Session(graph = graph) as sess :
        sess.run(tf.global_variables_initializer())
        y_, output_= sess.run([y, output],  feed_dict={inX_tf:[19,72],
                           dataset_tf: dataset,
                           labels_tf: labels})
        print(output_)

```

5. Section 6 excercise 
```python3
graph = tf.Graph()
with graph.as_default():
    xs = tf.placeholder(tf.float32, shape=(None,), name='x')
    y_true = tf.placeholder(tf.float32, shape=(None,), name='y_true')

    # 변수 초기화
    with tf.variable_scope('weights'):
        W = tf.Variable(tf.random.normal([1]),"W")
        b = tf.Variable(tf.zeros([1]),'b')
        tf.summary.scalar(name = 'W', tensor = W[0])
        tf.summary.scalar(name='b', tensor = b[0])

    # Model
    with tf.variable_scope("Linear_Regression"):
        y_pred = W * xs + b
    y_pred = tf.identity(y_pred, name="y_pred")
    
    # Loss Function
    with tf.variable_scope("losses"):
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        tf.summary.scalar(name="loss", tensor = loss)
        
    merged_all = tf.summary.merge_all()
    # Optimizer
    train_op = (tf.train
                .GradientDescentOptimizer(0.01)
                .minimize(loss))
    
log_dir = "./log/"
tbc = tensorboardcolab.TensorBoardColab(graph_path=log_dir)
writer = tf.summary.FileWriter(log_dir)

with graph.as_default():
    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, summary_values = sess.run([train_op, merged_all],
                                     feed_dict={
                                         xs:data_X,
                                         y_true:data_Y
                                     })
        writer.add_summary(summary_values, step)
```
