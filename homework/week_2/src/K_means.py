import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
xs = np.array([
    [3, 104],
    [2, 100],
    [1, 81],
    [101, 10],
    [99, 5],
    [98, 2],
])

ys = np.array(['Romance', 'Romance', 'Romance', 'Action', 'Action', 'Action'])

K = 2
target_x = np.array([25, 87])
# target_x 와 모든 데이터의 데이터의 거리를 구합니다 (L2)

# distance
dists = np.sqrt(np.sum(((target_x - xs) ** 2), axis=1))

# 데이터의 범위
min_feature_0, min_feature_1 = np.min(xs, axis=0)
max_feature_0, max_feature_1 = np.max(xs, axis=0)

# random 하게 k 개의 점 찍기
rand_feature_0 = npr.uniform(min_feature_0, max_feature_0, size=2)
rand_feature_1 = npr.uniform(min_feature_1, max_feature_1, size=2)
centroids = np.stack([rand_feature_0, rand_feature_1], axis=-1)

while True:
    dists = np.sqrt(np.sum((xs.reshape(-1, 1, 2) - centroids.reshape(1, -1, 2)) ** 2, axis=-1))
    pred = np.argmin(dists, axis=1)
    new_centroids = []
    for k_ind in range(K):
        centroid = np.mean(xs[np.where(pred == k_ind)[0]], axis=0)
        new_centroids.append(centroid)

    new_centroids = np.asarray(new_centroids)
    if np.all(np.asarray(new_centroids) == centroids):
        break
    else:
        centroids = new_centroids

plt.scatter(xs[:, 0], xs[:, 1], label='data')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', label='data')
plt.show()
