#1번 문제
```python3

def classify_knn(inX, dataset, labels, K):

    dataset_norm = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)

    inX_norm = (inX - dataset.mean(axis=0)) / dataset.std(axis=0) #input data도 dataset의 평균과 분산으로 scaling?

    dist = np.sum((dataset_norm - inX_norm)**2, axis= 1)

    #dist = np.sum((dataset - inX)**2, axis= 1)

    sorted_arg = dist.argsort()


    labels_K = labels[sorted_arg[:K]]


    la_ , inx_ = np.unique(labels_K, return_counts = True)

    print(la_[inx_.argmax()])


    pass

```

#2번문제
```python3
def cluster_kmedians(dataset, k):    
    
    dataset = np.asarray(dataset)
    
    min_0 , min_1 = dataset.min(axis = 0)
    max_0 , max_1 = dataset.max(axis = 0)
    
    rand_0 = np.random.uniform(min_0, max_0, size = 2)
    rand_1 = np.random.uniform(min_1, max_1, size = 2)
    
    centroids = np.stack([rand_0, rand_1], axis = 1) 
    
    
    cnt = 0
    while True:
        dist = np.sum(np.abs(dataset.reshape(-1,1,2) - centroids.reshape(-1,2,2)), axis = 2)

        distmin_idx = dist.argmin(axis=1)

        cent_= []
        for k_ in range(k):
            cen = np.median(dataset[np.where(distmin_idx == k_)], axis = 0)
            cent_.append(cen)
        
        cent_ = np.asarray(cent_)
        
        if( np.all(centroids == cent_)) :
            break
        else :
            centroids = cent_

    plt.scatter(dataset[:, 0], dataset[:, 1], label='data')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', label='data')
    plt.show()
    
    return 
```
