# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as cdist
from seaborn import heatmap

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy", allow_pickle=False)
small_dataset = np.load("data/small_dataset.npy")
small_labels = np.load("data/small_dataset_labels.npy").astype(int)
large_dataset = np.load("data/large_dataset.npy")
test = np.load("P2_Autograder_Data.npy")
true_labels = np.load("data/small_labels.npy")

print(pics.shape)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.mu = np.random.rand(self.K, large_dataset.shape[1])
        self.losses = []
            

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        self.x_class = np.zeros(X.shape[0])
        old_loss = np.inf
        new_loss = self.__calc_loss(X)
        self.losses.append(new_loss)
        while (old_loss - new_loss) > 5000:
            old_loss = new_loss
            for i in range(X.shape[0]):
                self.x_class[i] = self.__find_closest(X[i])
            for k in range(self.mu.shape[0]):
                total = np.zeros(large_dataset.shape[1])
                counter = 0
                for j, cond in enumerate(self.x_class == k):
                    if cond:
                        total += X[j]
                        counter += 1
                if counter != 0:
                    self.mu[k] = np.divide(total, counter)
            new_loss = self.__calc_loss(X)
            self.losses.append(new_loss)
        pass
    
    def __find_closest(self, x):
        dist = []
        for mu in self.mu:
            dist.append(np.linalg.norm(x - mu))
        return np.argmin(dist)
    
    def __calc_loss(self, X):
        loss = 0
        for n, x in enumerate(X):
            for k in range(self.mu.shape[0]):
                if (self.x_class[n] == k):
                    loss += np.matmul((x - self.mu[k]).T,(x - self.mu[k]))
        return loss
    
    def plot_loss(self):
        x = np.arange(len(self.losses))
        plt.figure()
        plt.plot(x, self.losses)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Objective Function')
        plt.title('K-Means Objective Function vs. Iterations')
        plt.savefig('2_1.png')
        pass
    
    def get_loss(self):
        return self.losses[len(self.losses)-1]
    
    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        means = np.zeros((self.K,28,28))
        for k, mu in enumerate(self.mu):
            means[k] = mu.reshape(28,28)
        return means
    
    def get_clusters(self):
        return self.x_class

K = 10
KMeansClassifier = KMeans(K)
KMeansClassifier.fit(large_dataset)
KMeansClassifier.plot_loss()

Ks = [5, 10, 20]
losses = np.zeros((3,5))
for i in range(5):
    for j, k in enumerate(Ks):
        Classifier = KMeans(k)
        Classifier.fit(large_dataset)
        losses[j][i] = Classifier.get_loss()
means = [np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])]
stds = [np.std(losses[0]), np.std(losses[1]), np.std(losses[2])]

plt.figure()
plt.bar([0,1,2],means,yerr=stds)
plt.xticks(ticks=[0,1,2],labels=['5','10','20'])
plt.xlabel('K')
plt.ylabel('Objective Function')
plt.title('K-Means Objective Value vs. Values of K')
plt.ylim(1.1e10,1.45e10)
plt.savefig('2_2.png')

plt.figure()
fig, axs = plt.subplots(10,3, figsize=(15, 45))
for j in range(10):
    axs[j,0].set_ylabel('Class '+str(j))
for i in range(3):
    KMeansClassifier = KMeans(K)
    KMeansClassifier.fit(large_dataset)
    imgs = KMeansClassifier.get_mean_images()
    axs[0,i].set_title('iteration '+str(i))
    for j in range(10):
        axs[j,i].imshow(imgs[j],cmap='Greys_r')
plt.savefig('2_3.png')

mean = np.mean(large_dataset, axis=0)
stds = np.std(large_dataset, axis=0)
stds[stds == 0] = 1
standardized = (large_dataset - mean) / stds
plt.figure()
fig, axs = plt.subplots(10,3,figsize=(15,45))
for j in range(10):
    axs[j,0].set_ylabel('Class '+str(j))
for i in range(3):
    KMeansClassifier = KMeans(K)
    KMeansClassifier.fit(standardized)
    imgs = KMeansClassifier.get_mean_images()
    axs[0,i].set_title('iteration '+str(i))
    for j in range(10):
        axs[j,i].imshow(imgs[j],cmap='Greys_r')
plt.savefig('2_4.png')

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(pics[0].reshape(28,28), cmap='Greys_r')
plt.show()


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.distances = []
    
    def fit(self, X):
        self.X = X
        self.clusters = np.arange(X.shape[0])
        self.dists = cdist(self.X, self.X, 'euclidean')
        while np.unique(self.clusters).shape[0] > 10:
            c1, c2, dist = self.__get_closest_clusters()
            self.distances.append(dist)
            self.__merge_cluster(c1, c2)
    
    def __merge_cluster(self, c1, c2):
        a = min(c1,c2)
        b = max(c1,c2)
        for i in np.nonzero(self.clusters == b)[0]:
            self.clusters[i] = a
    
    def __get_closest_clusters(self):
        unique = np.unique(self.clusters)
        best_dist = None
        best_clusters = None
        for i, cluster1 in enumerate(unique):
            for j, cluster2 in enumerate(unique):
                if i <= j:
                    continue
                else:
                    indices1 = np.nonzero(self.clusters == cluster1)
                    indices2 = np.nonzero(self.clusters == cluster2)
                    proposal = self.__calc_linkage_distances(indices1[0],indices2[0])
                    if (best_dist is None) or (proposal < best_dist):
                        best_dist = proposal
                        best_clusters = (cluster1, cluster2)
        return best_clusters[0], best_clusters[1], best_dist

    def __calc_linkage_distances(self, i1, i2):
        if self.linkage == 'min':
            dist = np.min(self.dists[i1,:][:,i2])
            return dist
        elif self.linkage == 'max':
            dist = np.max(self.dists[i1,:][:,i2])
            return dist
        elif self.linkage == 'centroid':
            centroid1 = np.mean(self.X[i1,:],axis=0)
            centroid2 = np.mean(self.X[i2,:],axis=0)
            return np.sqrt(np.sum((centroid1 - centroid2)**2))
    
    def get_clusters(self):
        return self.clusters
    
    def get_distances(self):
        return self.distances
    
    def get_mean_images(self):
        means = np.zeros((10,28,28))
        for i in range(10):
            total = np.zeros(self.X.shape[1])
            for n in np.where(self.clusters == np.unique(self.clusters)[i])[0]:
                total += self.X[n]
            means[i] = (total/len(np.where(self.clusters == np.unique(self.clusters)[i])[0])).reshape(28,28)
        return means
    
MinHAC = HAC('min')
MinHAC.fit(small_dataset)

MaxHAC = HAC('max')
MaxHAC.fit(small_dataset)

CentroidHAC = HAC('centroid')
CentroidHAC.fit(small_dataset)

plt.figure()
fig, axs = plt.subplots(10,3,figsize=(15,45))
imgs = MinHAC.get_mean_images()
for j in range(10):
    axs[j,0].imshow(imgs[j],cmap='Greys_r')
    axs[j,0].set(ylabel= 'Class ' + str(j))
imgs = MaxHAC.get_mean_images()
for j in range(10):
    axs[j,1].imshow(imgs[j],cmap='Greys_r')
imgs = CentroidHAC.get_mean_images()
for j in range(10):
    axs[j,2].imshow(imgs[j],cmap='Greys_r')
axs[0,0].set_title('Min HAC')
axs[0,1].set_title('Max HAC')
axs[0,2].set_title('Centroid HAC')
plt.savefig('2_5.png')

x = np.arange(len(MinHAC.get_distances()))
plt.plot(x,MinHAC.get_distances(),label="Min HAC")
plt.plot(x,MaxHAC.get_distances(),label="Max HAC")
plt.plot(x,CentroidHAC.get_distances(), label="Centroid HAC")
plt.legend()
plt.title('Iterations vs. Distance')
plt.xlabel('Total Number of merges completed')
plt.ylabel('Distance between most recently merged clusters')
plt.savefig('2_6.png')

plt.figure()
plt.bar(np.arange(10),np.unique(MinHAC.get_clusters(), return_counts = True)[1], label='Min HAC')
plt.bar(np.arange(10),np.unique(MaxHAC.get_clusters(), return_counts = True)[1], label='Max HAC')
plt.title('Distribution of Clusters')
plt.xlabel('Cluster index')
plt.ylabel('Number of images in cluster')
plt.legend()
plt.savefig('2_7.png')

def conf_mat(trues, preds):
    mat = np.zeros((np.unique(trues).shape[0],np.unique(trues).shape[0]))
    for i in range(np.unique(trues).shape[0]):
        for j in range(np.unique(trues).shape[0]):
            true_clusters = np.nonzero(trues == j)[0]
            pred_clusters = np.nonzero(preds == i)[0]
            for t in true_clusters:
                if np.any(pred_clusters == t):
                    mat[i][j] += 1
    return mat

K = 10
KMeans_small = KMeans(K)
KMeans_small.fit(small_dataset)
KMeans_clusters = KMeans_small.get_clusters()
for i, c in enumerate(np.unique(KMeans_clusters)):
    for n in np.where(KMeans_clusters==c):
        KMeans_clusters[n] = i
plt.figure()
heatmap(conf_mat(true_labels,KMeans_clusters),annot=True)
plt.savefig('2_8_KMeans.png')

min_clusters = MinHAC.get_clusters()
for i, c in enumerate(np.unique(min_clusters)):
    for n in np.where(min_clusters==c):
        min_clusters[n] = i
plt.figure()
heatmap(conf_mat(true_labels,min_clusters), annot=True)
plt.savefig('2_8_MinHAC.png')

max_clusters = MaxHAC.get_clusters()
for i, c in enumerate(np.unique(max_clusters)):
    for n in np.where(max_clusters==c):
        max_clusters[n] = i
plt.figure()
heatmap(conf_mat(true_labels,max_clusters), annot=True)
plt.savefig('2_8_MaxHAC.png')

centroid_clusters = CentroidHAC.get_clusters()
for i, c in enumerate(np.unique(centroid_clusters)):
    for n in np.where(centroid_clusters==c):
        centroid_clusters[n] = i
plt.figure()
heatmap(conf_mat(true_labels,centroid_clusters),annot=True)
plt.savefig('2_8_CentroidHAC.png')