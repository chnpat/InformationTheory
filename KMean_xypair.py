import matplotlib.pyplot as plt
import numpy as np
import math


class KMeans:

    def __init__(self, k=2, tol=0.001, max_iteration=500):
        self.k = k
        self.tol = tol
        self.max_iteration = max_iteration
        self.centroids = {}
        self.clusters = {}

    def compute(self, sample):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = sample[i]

        for j in range(self.max_iteration):

            self.clusters = {}

            for k in range(self.k):
                self.clusters[k] = []

            for point in sample:
                distance = [np.linalg.norm(point - self.centroids[cen]) for cen in self.centroids]
                cluster = distance.index(min(distance))

                self.clusters[cluster].append(point)

            prev_centroid = dict(self.centroids)

            for clust in self.clusters:
                self.centroids[clust] = np.average(self.clusters[clust], axis=0)

            optimized = True

            for cent in self.centroids:
                original_centroids = prev_centroid[cent]
                current_centroids = self.centroids[cent]
                s = np.sum((current_centroids - original_centroids) / original_centroids * 100.0)
                if s > self.tol:
                    optimized = False

            if optimized:
                break
            print("Iteration = " + str(j))


# Variables
M = 1000
K = 8
np.random.seed(1920431)
x = np.random.normal(loc=0, scale=1, size=(M, 2))

km = KMeans(K, tol=0.001, max_iteration=1000)
km.compute(x)

colorarr = []
for i in range(0, K):
    rand_arr = np.random.random((1, 3))
    colorarr.append((rand_arr[0][0], rand_arr[0][1], rand_arr[0][2], 1.0))
colors = np.array(colorarr)

for cen in km.centroids:
    plt.scatter(km.centroids[cen][0], km.centroids[cen][1], marker="o", color="k", s=1, linewidths=5)

MSE_x = 0
MSE_y = 0
for cluster in km.clusters:

    color = colors[cluster]
    for sample in km.clusters[cluster]:
        plt.scatter(sample[0], sample[1], color=color, s=1, linewidths=5)
        MSE_x = MSE_x + math.pow((sample[0] - km.centroids[cluster][0]), 2)
        MSE_y = MSE_y + math.pow((sample[1] - km.centroids[cluster][1]), 2)

MSE_x = MSE_x / M
MSE_y = MSE_y / M

print("MSE = (" + str(MSE_x) + "," + str(MSE_y) + ")")
plt.show()
