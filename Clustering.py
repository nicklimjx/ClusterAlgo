import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from tqdm import tqdm
from dtaidistance import dtw

# holy fucking shit the amount of dependencies just to change the distance metric to dtw is fucking nuts lmao'
# also dtw.fastdistance is so mich faster than fastdtw because of c

class KMeansDTW():
    def __init__(self, k: int = 8, max_iter: int = 3000, tol: float = 0.001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        
    def create_clusters(self, data: np.ndarray):
        # Initialize centroids randomly
        rand_idx = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[rand_idx]
        
        for _ in tqdm(range(self.max_iter)):
            self.classifications = [[] for _ in range(self.k)]
            
            # Precompute distances between each datapoint and each centroid
            # This step is assumed to be the optimized part; depending on the fastdtw implementation details
            # You might need to manually loop through data and centroids if fastdtw cannot be vectorized directly
            for i, datapoint in enumerate(data):
                distances = np.array([dtw.distance_fast(centroid, datapoint) for centroid in self.centroids])
                closest_centroid_idx = np.argmin(distances)
                self.classifications[closest_centroid_idx].append(datapoint)
            
            prev_centroids = np.copy(self.centroids)
            for i, classification in enumerate(self.classifications):
                # Efficiently compute new centroids
                if classification:  # Check if classification is not empty
                    self.centroids[i] = np.mean(classification, axis=0)
            
            # Check for convergence
            optimised_flag = True
            for i in range(self.k):
                diff = np.linalg.norm(prev_centroids[i] - self.centroids[i])
                if diff >= self.tol:
                    optimised_flag = False
                    break
            
            if optimised_flag:
                break

    def elbow_method(self):
        total_var = 0
        for i, centroid in enumerate(self.centroids):
            for datapoint in self.classifications[i]:
                total_var += dtw.distance_fast(centroid, datapoint)
        return total_var
    
    def display_clusters(self):
        for i, cluster in enumerate(self.classifications, start = 1):
            plt.figure(figsize=(3, 1.5))
            for series in cluster:
                plt.plot(series)

        plt.title(f'Cluster {i} Time Series')
        plt.show()


class HierarchDTW():
    def __init__(self, data: np.ndarray):
        self.data = data
        self.linkages_matrix = None
    
    def calc_dist_matrix(self):
        num_datapoints = self.data.shape[0]
        self.distance_matrix = np.zeros((num_datapoints, num_datapoints))
        for i in range(num_datapoints):
            for j in range(i + 1, num_datapoints):
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = dtw.distance_fast(self.data[i], self.data[j])
        
    def cluster(self, method = 'ward'):
        if self.linkages_matrix is None:
            self.calc_dist_matrix()
        self.linkages_matrix = linkage(squareform(self.distance_matrix), method = method)

    def plot_dendrogram(self):
        if self.linkages_matrix is None:
            print('not clustered yet')
        else:
            plt.figure(figsize = (10, 7))
            dendrogram(self.linkages_matrix)
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.show()

    def display_clusters(self, max_clusters):
        print(self.linkages_matrix.shape)
        cluster_labels = fcluster(self.linkages_matrix, t = max_clusters, criterion = 'maxclust')
        print(cluster_labels, type(cluster_labels))
        for i in range(1,  len(np.unique(cluster_labels)) + 1):
            idx = np.where(cluster_labels == i)
            plt.plot(self.data[idx])
            plt.title(f'Cluster {i} for hierarchical')
            plt.show()

def split_time_series(series, window: int, slide: int):
    split = []
    series = zscore(series)
    for i in range(0, int(len(series) * 0.7) - window, slide): #magic number, splitting into train
        split.append(list(series.iloc[i:i+window]))
    return np.array(split)

train_df = pd.read_csv('TrainData.csv', delimiter=',')
tester = split_time_series(train_df['close'], 36, 18) # magic number here, should be fine tuned??

elbows = []
for i in range(1, 13):
    my_KMeans = KMeansDTW(i)
    my_KMeans.create_clusters(tester)
    elbows.append(my_KMeans.elbow_method())

my_Hierarch = HierarchDTW(tester)
my_Hierarch.cluster()
my_Hierarch.display_clusters(max_clusters= 12)
# look this is a horrendously non rigorous way to do it but pretty much by visual inspection, i've decided that 6, 7 and 12 are outlier clusters
# this means that we cut them out, we should decide on a more rigorous way to do this, maybe based on how high up the dendrogram the connection is