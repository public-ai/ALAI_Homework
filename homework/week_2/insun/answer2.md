# Week 2. Homework 2. Numpy Basis_K Median 군집화

```python3
%matplotlib inline
import numpy as np

import matplotlib.pyplot as plt
```

### 이상값이 존재하는 데이터셋
```python3
# 예제 데이터 셋 
!wget https://s3.ap-northeast-2.amazonaws.com/pai-datasets/alai-deeplearning/kmedian_dataset.csv
dataset = np.loadtxt("./kmedian_dataset.csv",delimiter=',')
```

### K-Median 알고리즘 구현하기
```python3
def cluster_kmedians(dataset, k):    
    ##########
    # CODE HERE!
    # 위의 결과가 COMEDY가 나올 수있도록 코드를 수정해 주세요!
    ##########     
    
    # (1) 중심점 초기화
    min_x = dataset[:,0].min()
    max_x = dataset[:,0].max() 
    min_y = dataset[:,1].min()
    max_y = dataset[:,1].max() 

    center_x = np.random.uniform(low=min_x, high=max_x, size=k)
    center_y = np.random.uniform(low=min_y, high=max_y, size=k)
    centroids = np.stack([center_x,center_y],axis=-1)
    
    # (2) ~ (5) 순회
    num_data = dataset.shape[0]
    cluster_per_point = np.zeros((num_data)) # 각 점 별 군집

    counter = 0
    while True:
        prev_cluster_per_point = cluster_per_point
        
        # (2) 거리 계산
        diff_mat = (centroids.reshape(-1,1,2) - dataset.reshape(1,-1,2))
        
        ############################# 수정 #############################
        # dists = np.sqrt((diff_mat**2).sum(axis=-1))
        dists = np.abs(diff_mat).sum(axis=-1)
        ################################################################
        
        # (3) 각 데이터를 거리가 가장 가까운 군집으로 할당
        cluster_per_point = dists.argmin(axis=0)
        
        # (4) 각 군집 별 점들의 평균을 계산 후, 군집의 중심점을 다시 계산
        for i in range(k):
            ############################# 수정 #############################
            # centroids[i] = dataset[cluster_per_point==i].mean(axis=0)
            mask = dataset[cluster_per_point==i]
            centroids[i] = np.median(mask, axis=0)
            ################################################################
            
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
```


```python3
# 아래와 같은 결과가 나오면 됩니다
cluster_kmedians(dataset, 2)
```