# 22ID60R20: Gajendra Singh
# Project Code: OCHC-DS 
# Project Title: Obesity Data Categorization using Single Linkage Divisive (Top-Down) Clustering Technique



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
import random

# Import the data
df = pd.read_csv('obesity.csv')

######################## Data Preprocessing ########################

# df.head()
# df.info()
# df.shape
# df.describe()

#print(df.isna().sum().sort_values(ascending = False))  # No NA value found after printing

df=df.drop_duplicates() # Delete all the duplicate rows considering all column wise.
print("Final shape of data after preprocessing", df.shape, '\n')   # final shape of datadata




df['CAEC'].value_counts()                                      # converting categorical data to numeric data
df = pd.get_dummies(df, columns=['CAEC'])

df['Gender'].value_counts()                                    # converting categorical data to numeric data
df['Gender'].replace(['Male', 'Female'],[1, 0], inplace=True)

df['family_history_with_overweight'].value_counts()            # converting categorical data to numeric data
df['family_history_with_overweight'].replace(['yes', 'no'],[1, 0], inplace=True)

df['FAVC'].value_counts()                                      # converting categorical data to numeric data
df['FAVC'].replace(['yes', 'no'],[1, 0], inplace=True)

df['SMOKE'].value_counts()                                     # converting categorical data to numeric data
df['SMOKE'].replace(['yes', 'no'],[1, 0], inplace=True)

df['SCC'].value_counts()                                       # converting categorical data to numeric data
df['SCC'].replace(['yes', 'no'],[1, 0], inplace=True)

df['CALC'].value_counts()                                      # converting categorical data to numeric data
df = pd.get_dummies(df, columns=['CALC'])

df['MTRANS'].value_counts()                                    # converting categorical data to numeric data
df = pd.get_dummies(df, columns=['MTRANS'])



                     ############################### K-Mean Clustering ##################################

# define the class for KMean clustering
class KMeans:
    # variable assign
    def __init__(self, n_clusters, max_iter = 20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
                
    # function to make cluster of data
    def fit_predict(self,X):
        random_index = random.sample(range(0,X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
      
        for i in range(self.max_iter):
            #assign cluster
            cluster_group = self.assign_cluster(X)
            old_centroids = self.centroids
           
            #move centroid
            self.centroids = self.move_centroids(X,cluster_group)
            
        
            #check finish
            if (old_centroids == self.centroids).all():
                break
            
        return cluster_group
        
    # Assign cluster according to cosine similarity            
    def assign_cluster(self,X):
        cluster_group = []
        similarity = []
        
        for row in X:
            for centroid in self.centroids:
                similarity.append(np.dot(row,centroid)/(norm(row)*norm(centroid)))
            max_similarity = max(similarity)
            index_pos = similarity.index(max_similarity)
            cluster_group.append(index_pos)
            similarity.clear()
        
        return np.array(cluster_group)
    
    # Assign new centroid after calculating mean of clustered data
    def move_centroids(self,X,cluster_group):
        new_centroids = []
        
        cluster_type = np.unique(cluster_group)
        
        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))
       
        return np.array(new_centroids)
    
    # function to calculate euclidean distance
    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    # function to calculate silhouete coefficient of clustered data
    def silhouette_coefficient(self,X, cluster_group):
    # Initialize empty arrays to store the silhouette coefficients and distances
        silhouette_coeffs = np.zeros(X.shape[0])
        distances = np.zeros((X.shape[0], X.shape[0]))

        # Calculate the distances between all pairs of samples
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[0]):
                distances[i, j] = distances[j, i] = self.euclidean_distance(X[i], X[j])

        # Calculate the silhouette coefficient for each sample
        for i in range(X.shape[0]):
            # Calculate the mean distance to other samples in the same cluster
            a = np.mean(distances[i, cluster_group == cluster_group[i]])
            
            # Calculate the mean distance to other samples in the closest other cluster
            b = np.min([np.mean(distances[i, cluster_group == j]) for j in range(cluster_group.max()+1) if j != cluster_group[i]])
            
            # Calculate the silhouette coefficient
            if a < b:
                silhouette_coeffs[i] = 1 - a/b
            elif a > b:
                silhouette_coeffs[i] = b/a - 1
            else:
                silhouette_coeffs[i] = 0

        # Return the mean silhouette coefficient for all samples
        return np.mean(silhouette_coeffs)


# X from the given dataset
X = df.iloc[:,:].values

# Taking different k for different no of cluster
different_k = [3,4,5,6]

# list to store Silhouette cofficient for different k
different_sil = []

for t in different_k:
    km = KMeans(t, max_iter = 20)
    cls = km.fit_predict(X)

    clusters_KMean = [[] for _ in range(t)]
    for i in range(len(cls)):
        clusters_KMean[cls[i]].append(i)
    sil_coeff = km.silhouette_coefficient(X, cls)
    print(f"Silhouette coefficient for k = {t} after KMean clustering, is {sil_coeff}")
    different_sil.append(sil_coeff)

    fp = open(f"kmeans_k{t}.txt", "w")         # file to store the clustered data
    
    for cluster in clusters_KMean:
    
        s = ",".join(str(i) for i in cluster) + "\n"      
        fp.write(s)                              # store the clustered data into file 
    
    fp.close()


# optimum k for which silhouette coeff is maximum    
max_sil=0
a = 2
for s in different_sil:                               
    a = a + 1
    
    if s>max_sil:
        optimum_k = a
        max_sil = s  
        print("\n")                        
        print(f"Optimum k (no of cluster) is {optimum_k}, having silhouette coefficient {max_sil}" )


# call the class KMeans
km = KMeans(n_clusters = optimum_k, max_iter = 20)
cls_k3 = km.fit_predict(X)

# Makes the separate list for individual clustered data
clusters_KM_k3 = [[] for _ in range(optimum_k)]
for i in range(len(cls_k3)):
    clusters_KM_k3[cls_k3[i]].append(i)

        



     ############################  Single Linkage Divisive (Top-Down) Clustering   #################################


                     
# Perform divisive clustering on the input data using cosine similarity.

import numpy as np

def divisive_clustering(data, k):
    
    # Convert data to unit vectors
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    
    # Initialize with all data points in one cluster
    clusters_Hi = [list(range(data.shape[0]))]
    
    while len(clusters_Hi) < k:
        # Find the cluster with the largest SSE
        sses = [np.sum(np.var(data[cluster], axis=0)) for cluster in clusters_Hi]
        largest_sse_idx = np.argmax(sses)
        
        # Split the largest cluster in two using cosine similarity
        largest_cluster = np.array(clusters_Hi.pop(largest_sse_idx))
        similarities = np.dot(data[largest_cluster], data[largest_cluster].T)
        np.fill_diagonal(similarities, -1)
        max_similarity_idx = np.argmax(similarities)
        cluster1_idx, cluster2_idx = np.unravel_index(max_similarity_idx, similarities.shape)
        cluster1 = [largest_cluster[cluster1_idx]]
        cluster2 = [largest_cluster[cluster2_idx]]
        for i, point in enumerate(largest_cluster):
            if i not in {cluster1_idx, cluster2_idx}:
                if similarities[i, cluster1_idx] > similarities[i, cluster2_idx]:
                    cluster1.append(point)
                else:
                    cluster2.append(point)
        
        # Add the new clusters to the list of clusters
        clusters_Hi.append(cluster1)
        clusters_Hi.append(cluster2)
    
    return clusters_Hi            # Returns a list of k clusters, each represented as a list of indices into the data array.

# call a function as cluster_Hi
cluster_Hi = divisive_clustering(X, optimum_k)

def silhouette_coefficient(data, clusters):
    """
    Calculates the silhouette coefficient for the given data and clustering.
    
    Args:
    data (np.ndarray): The data points to cluster.
    clusters (list): A list of lists, where each inner list contains the indices of the data points
                     in the corresponding cluster.
                     
    Returns:
    float: The average silhouette coefficient for the given clustering.
    """

    # Calculate the similarity matrix
    similarities = np.dot(data, data.T)
    
    # Iterate over each data point
    num_points = data.shape[0]
    silhouette_coeffs = np.zeros(num_points)
    for i in range(num_points):
        # Calculate the average distance to other points in the same cluster
        cluster = None
        for j, c in enumerate(clusters):
            if i in c:
                cluster = c
                break
        if cluster is None:
            continue
        avg_intra_dist = np.mean(similarities[i, cluster])
        
        # Calculate the average distance to other points in the nearest neighboring cluster
        nearest_cluster = None
        nearest_dist = np.inf
        for j, c in enumerate(clusters):
            if c == cluster:
                continue
            dist = np.mean(similarities[i, c])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_cluster = c
        
        # Calculate the silhouette coefficient for this point
        if nearest_cluster is None:
            silhouette_coeffs[i] = 0
        else:
            silhouette_coeffs[i] = (nearest_dist - avg_intra_dist) / max(avg_intra_dist, nearest_dist)
    
    # Return the average silhouette coefficient for all points
    return np.mean(silhouette_coeffs)


sil_Hi = silhouette_coefficient(X, cluster_Hi)
print(f"\n silhouette coefficient after divisive clustering for optimum cluseters is {sil_Hi}.")

# store the final cluster information considering the optimal number of clusters
fp = open(f"divisive_K{optimum_k}.txt", "w")         # file to store the clustered data
    
for cluster in cluster_Hi:
    
    s = ",".join(str(i) for i in cluster) + "\n"      
    fp.write(s)                                 # store the clustered data into file 
    
fp.close()



         ##################################  Jaccard similarity  #######################################


# function for finding the similarity between two binary vectors

def jaccard_binary(x,y):
    li = []
    li1 = []
    for i in range(len(x)):
        for j in range(len(y)):
            b = 0
            # intersection = np.logical_and(x[i], y[j])
            # union = np.logical_or(x[i], y[j])
            # similarity = intersection.sum() / float(union.sum())

            intersection = len(list(set(x[i]).intersection(y[j])))
            union = (len(x[i]) + len(y[j])) - intersection
            similarity = float(intersection) / union


            li.append(similarity)
        b = max(li)
        li1.append(b)
    return li1

# call the function as out
out = jaccard_binary(clusters_KM_k3,cluster_Hi)
print('\n')
print(f"Jaccard similarity between corresponding sets of both the cases is {out}.")




