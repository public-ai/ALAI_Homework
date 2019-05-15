<pre><code>
def classify_knn(inX, dataset, labels, K):
    #dataset을 표준점수로
    dataset = (dataset - np.mean(dataset, axis=0))/np.std(dataset, axis=0)
    
    dists = np.sqrt(np.sum((inX-dataset)**2,axis=1))
    
    sorted_index = dists.argsort()
    
    sorted_labels = labels[sorted_index]
    k_nearst_labels = sorted_labels[:K]
    
    _labels, count_labels = np.unique(k_nearst_labels, return_counts=True)
    
    return _labels[count_labels.argmax()]
    
    pass
</code></pre>


<pre><code>
def getMedian(x):
    len_x = len(x)
    if len_x == 0:
        return 0
    elif len_x%2==1 :
        return x[int((len_x-1)/2)]
    else:
        return (x[int((len_x-1)/2)]+x[int((len_x-1)/2+1)])/2

def cluster_kmedians(dataset, k):    
    #1. 중심점 초기화
    min_x = dataset[:,0].min()
    max_x = dataset[:,0].max()
    min_y = dataset[:,1].min()
    max_y = dataset[:,1].max()
    
    center_x = np.random.uniform(low=min_x, high=max_x, size=k)
    center_y = np.random.uniform(low=min_y, high=max_y, size=k)
    centroids = np.stack([center_x, center_y], axis=-1) #k개의 군집의 좌표
    
    num_data = dataset.shape[0]
    cluster_per_point = np.zeros((num_data)) #각 데이터셋 점 별 군집
    
    counter = 0
    
    while True:
        prev_cluster_per_point = cluster_per_point
        
        diff_mat = (centroids.reshape(-1,1,2) - dataset.reshape(1,-1,2))
        dists = np.sqrt(abs(diff_mat).sum(axis=-1))
        cluster_per_point = dists.argmin(axis=0)
        
        for i in range(k):
            target_point = dataset[cluster_per_point==i]
            sort_target_point_x = np.sort(target_point[:,0])
            sort_target_point_y = np.sort(target_point[:,1])
            centroids[i] = [getMedian(sort_target_point_x), getMedian(sort_target_point_y)]
            
        if np.all(prev_cluster_per_point == cluster_per_point):
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
    
    return centroids
</code></pre>