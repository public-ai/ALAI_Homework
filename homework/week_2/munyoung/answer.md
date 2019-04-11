### 문제1
~~~
dataset = np.array([
    [120, 3],
    [105, 2],
    [25, 12],
    [32, 15],
    [17, 9],
    [98, 5],
    [130, 1],
    [0, 16],
    [40, 20],
    [100, 10]
])

labels = np.array(["comedy", "comedy", "drama",
                   "drama","drama","comedy","comedy",
                   "drama","drama","comedy"])

def classify_knn(inX, dataset, labels, K):
    # (1) 우리가 분류항목을 알고자 하는 점 (inX)와 
    # 알고 있는 점들(dataset)과의 모든 점 거리를 계산
    min_value = np.array([-10, 0])
    max_value = np.array([40, 140])
    dataset = (dataset - min_value) / (max_value - min_value)
    inX = (inX - min_value) / (max_value - min_value)
    dists = np.sqrt(np.sum((inX-dataset)**2, axis=1))
    
    
    # (2) 오름 차순으로 거리의 길이를 정렬
    sorted_index = dists.argsort()
    
    # (3) inX와의 거리가 가장짧은 K개의 아이템 추출
    sorted_labels = labels[sorted_index]
    K_nearest_labels = sorted_labels[:K]
    
    # (4) K개의 아이템에서 가장 많은 분류 항목 찾기
    _labels, count_labels = np.unique(K_nearest_labels,
          return_counts=True)
    # (5) 해당 항목 반환
    return _labels[count_labels.argmax()]

print(classify_knn(inX, dataset, labels, 4))
~~~

### 문제 2
~~~
def cluster_kmedians(dataset, k):
    # (1) 중심점 초기화
    min_x = dataset[:,0].min()
    max_x = dataset[:,0].max() 
    min_y = dataset[:,1].min()
    max_y = dataset[:,1].max() 
    
    center_x = np.random.uniform(low=min_x, high=max_x, size=k)
    center_y = np.random.uniform(low=min_y, high=max_y, size=k)
    centroids = np.stack([center_x,center_y],axis=-1)
    
    num_data = dataset.shape[0]
    cluster_per_point = np.zeros((num_data))

    counter = 0
    while True:
        print("run", counter)
        pre_cluster_per_point = cluster_per_point
        diff_mat = (centroids.reshape(-1,1,2) - dataset.reshape(1,-1,2))
        dists = np.sum(np.absolute(diff_mat), axis = -1)
        cluster_per_point = dists.argmin(axis = 0)
        
        for i in range(k):
            centroids[i][0] = np.median(dataset[cluster_per_point == i, 0])
            centroids[i][1] = np.median(dataset[cluster_per_point == i, 1])
            
        print("pre_cluster_per_point", pre_cluster_per_point)
        print("cluster_per_point", cluster_per_point)
        if np.all(pre_cluster_per_point == cluster_per_point):
            break
        
        counter += 1
        plt.title("{}th Distribution of Dataset".format(counter))
        for idx, color in enumerate(['r','g','b','y']):
            mask = (cluster_per_point==idx)
            plt.scatter(dataset[mask,0],dataset[mask,1],
                        label='dataset', c=color)
            plt.scatter(centroids[:,0],centroids[:,1],
                        s=200, label="centroid", marker='+')
        plt.show()
~~~
