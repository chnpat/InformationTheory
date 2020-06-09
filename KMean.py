###########################################
# KMean.py
###########################################
# Developed by: PATTIYANON, Charnon (1920431)
# Developed on: June 8th, 2020
###########################################
import math

import matplotlib.pyplot as plt
import numpy as np


###########################################


# "KMeans" class for computing K-Means algorithm
class KMeans:
    # Class constructor which initiates all local attributes
    def __init__(self, k=2, tol=0.001, max_iteration=500):
        self.k = k
        self.tol = tol
        self.max_iteration = max_iteration
        self.centroids = {}
        self.clusters = {}

    # A method to compute K-means by following the predefined algorithm
    def compute(self, sample):

        self.centroids = {}
        np.random.seed()

        # Initialization Step: Randomly select samples to be centroids
        for i in range(self.k):
            ind = np.random.randint(len(sample))
            self.centroids[i] = sample[ind]

        for j in range(self.max_iteration):

            self.clusters = {}

            for k in range(self.k):
                self.clusters[k] = []

            # Assignment Step: Assign each data to a cluster based on the distance
            # between the value and the centroid.
            for point in sample:
                distance = [np.linalg.norm(point - self.centroids[cen]) for cen in self.centroids]
                cluster = distance.index(min(distance))

                self.clusters[cluster].append(point)

            prev_centroid = dict(self.centroids)

            # Update Step: Update the centroid by the average of the data in the cluster
            for clust in self.clusters:
                self.centroids[clust] = np.average(self.clusters[clust], axis=0)

            # Condition to check whether the updated centroids are changed or not
            optimized = True

            for cent in self.centroids:
                original_centroids = prev_centroid[cent]
                current_centroids = self.centroids[cent]
                s = np.sum((current_centroids - original_centroids) / original_centroids * 100.0)
                if s > self.tol:
                    optimized = False

            if optimized:
                break


###########################################


# "RateDistortionComputation" Class which is a utility class for computing
# values in Rate-Distortion theory
class RateDistortionComputation:
    # Class constructor which initiates all local attributes
    def __init__(self):
        pass

    # A method to compute px for the values of gaussian distribution
    def compute_px(self, x, mean, variance):
        px = []
        for i in x:
            a = 1 / (math.sqrt(2 * math.pi * variance))
            b = -(math.pow((i - mean), 2) / 2 * variance)
            px.append(a * math.pow(math.e, b))
        return [p / np.sum(px) for p in px]

    # A method to compute the mean square error (or expected distortion) of the K-means operation
    def compute_mse(self, clusters, centroids, M):
        MSE = 0
        for c in clusters:
            for sample in clusters[c]:
                dist = math.pow(sample - centroids[c], 2)
                MSE = MSE + dist
        return MSE / M

    # A method to compute the rate-distortion function
    def compute_rate(self, D, variance):
        if D < variance:
            return 0.5 * math.log(variance / D, 2)
        else:
            return 0


###########################################
# Variables
M = 1000
K_set = [2, 4, 8]
mean = 0
variance = 5
###########################################
# Local variables initiation
mse = 0
R = 0
R_list = []
D_list = []
np.random.seed(1920431)

# Generate random sample from Gaussian distribution
x = np.random.normal(loc=mean, scale=math.sqrt(variance), size=M)

rd = RateDistortionComputation()

# Iterate over different K values
for K in K_set:
    print("-----------------------")
    print("K = " + str(K))
    print("-----------------------")
    km = KMeans(K, tol=0, max_iteration=1000)

    # Do the K-Means algorithm for several time to find the best (least) MSE
    for a in range(0, 10):
        km.compute(x)
        px = rd.compute_px(x, mean, variance)
        m = rd.compute_mse(km.clusters, km.centroids, M)
        if a == 0:
            mse = m
        if m < mse:
            mse = m
        R = rd.compute_rate(mse, variance)

    # Print out the result for each K value
    print("Mean = " + str(mean))
    print("Variance = " + str(variance))
    print("MSE (D) = " + str(round(mse, 4)))
    print("R = " + str(round(R, 4)))
    R_list.append(R)
    D_list.append(mse)

# Plot the rate-distortion function
axis_x = np.linspace(0.0001, variance, 100)
axis_y = [rd.compute_rate(b, variance) for b in axis_x]
plt.plot(axis_x, axis_y)
plt.xlim(0, variance)
plt.ylim(0, np.max(axis_y))
plt.ylabel("Rate R(D)")
plt.xlabel("Distortion D")

# Annotate all R,D pair for K = 2, 4, and 8 respectively
for i in range(0, len(K_set)):
    plt.annotate(". (R,D) = (" + str(round(R_list[i], 4)) + ", " + str(round(D_list[i], 4)) + "), K=" + str(K_set[i]),
                 (D_list[i], R_list[i]))

# Show the plot
plt.show()
