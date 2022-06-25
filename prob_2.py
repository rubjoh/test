import numpy as np
import matplotlib.pyplot as plt





data = np.genfromtxt(r"frey-faces.csv", delimiter=" ")

#For printing the pictures later in the problem
image_file = (r"frey-faces.csv")


class k_means:

    def __init__(self, k, max_iterations, dataset):

        self.k = k
        self.max_iterations = max_iterations

        self.dataset = dataset
        self.num_datapoints, self.num_features = dataset.shape
    
        

    def initialize_centroids(self):
        '''Method for randomly initializing the centriods'''

        #Creating an array of zeros with shape k * features
        centroids = [[] for k in range(self.k)]

        for i in range(self.k):

            initial_centroids = self.dataset[np.random.choice(range(self.num_datapoints))]
            centroids[i] = initial_centroids
        
        return centroids


    def find_distance(self, point1, point2):
        '''Method for finding the euclidean distance between two points'''

        distance = np.sqrt(np.sum((point1 - point2)**2, axis=1))

        return distance


    def clusters(self, centroids):
        '''Method for assigning datapoints to the clusters'''

        clusters = [[] for centroid in range(self.k)]

        for index, datapoint in enumerate(self.dataset):
            
            #Finding the index for the centroid that is closest to the datapoint
            min_dist_to_centroid = np.argmin(self.find_distance(datapoint, centroids))

            #Stores the index of the datapoints that is assigned to the respective centroids
            clusters[min_dist_to_centroid].append(index)

        return clusters
        
    
    def update_centroids(self, clusters):

        centroids = np.zeros((self.k, self.num_features))

        for index, cluster in enumerate(clusters):

            mean_cluster_val = np.mean(self.dataset[cluster], axis=0)

            centroids[index] = mean_cluster_val

        return centroids


    def fit(self):

        centroids = self.initialize_centroids()

        for i in range(self.max_iterations):

            clusters = self.clusters(centroids)

            former_centroid = centroids

            centroids = self.update_centroids(clusters)

            if not (former_centroid - centroids).any():
                print("Centroids have converged, number of iterations:", i)
                break

        return clusters, centroids



def find_random_dp(K, dataset, number_of_random_dp, max_iterations=300):

    initialize = k_means(k=K, dataset=dataset, max_iterations=max_iterations)

    clusters, centroids = initialize.fit()

    #Pick random datapoints from each cluster
    random_dp = []

    for i in range(K):

        random_index = np.random.choice(clusters[i], size=number_of_random_dp)

        random_dp.append(random_index)

    return random_dp, centroids



def sorting(k, rand_clust_idx, centroid_idx):

    for i in range(k):

        plotting = plot_images(data, rand_clust_idx[i], centroid_idx[i])

    return plotting


def plot_images(dataset, cluster_point_index, centroid):

    one_image = dataset[cluster_point_index[0]]
    two_image = dataset[cluster_point_index[1]]
    three_image = dataset[cluster_point_index[2]]
    four_image = dataset[cluster_point_index[3]]

    reshape_one = one_image.reshape((28,20))
    reshape_two = two_image.reshape((28,20))
    reshape_three = three_image.reshape((28,20))
    reshape_four = four_image.reshape((28,20))

    reshape_centroid = centroid.reshape((28, 20))

    figure, ax = plt.subplots(1, 5)

    #Image centroid:
    ax[0].imshow(reshape_centroid, cmap='gray')
    ax[0].set_title('Centroid')
    ax[0].set_axis_off()
    
    #Images for the clusters
    ax[1].imshow(reshape_one, cmap='gray')
    ax[1].set_title(f'Index: {cluster_point_index[0]}')
    ax[1].set_axis_off()
    ax[2].imshow(reshape_two, cmap='gray')
    ax[2].set_title(f'Index: {cluster_point_index[1]}')
    ax[2].set_axis_off()
    ax[3].imshow(reshape_three, cmap='gray')
    ax[3].set_title(f'Index: {cluster_point_index[2]}')
    ax[3].set_axis_off()
    ax[4].imshow(reshape_four, cmap='gray')
    ax[4].set_title(f'Index: {cluster_point_index[3]}')
    ax[4].set_axis_off()

    plt.tight_layout()
    plt.show()



print('''For two centroids:''')
rand_clust_index_2, centroid_2 = find_random_dp(K=2, dataset=data, number_of_random_dp=4, max_iterations=300)

print('''
For four centroids:''')
rand_clust_index_4, centroid_4 = find_random_dp(K=4, dataset=data, number_of_random_dp=4, max_iterations=300)

print('''
For ten centroids:''')
rand_clust_index_10, centroid_10 = find_random_dp(K=10, dataset=data, number_of_random_dp=4, max_iterations=300)


#Printing the images of the clusters with their respective centroid
sorting(2, rand_clust_index_2, centroid_2)
sorting(4, rand_clust_index_4, centroid_4)
sorting(10, rand_clust_index_10, centroid_10)







    

