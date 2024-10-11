from csv import DictReader
import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    res = []
    f = open(filepath, "r")
    csv_reader = DictReader(f)
    for row in csv_reader:
        res.append(dict(row))

    return res


def calc_features(row):
    features = numpy.zeros(6, dtype=numpy.float64)

    features[0] = row['Population']
    features[1] = row['Net migration']
    features[2] = row['GDP ($ per capita)']
    features[3] = row['Literacy (%)']
    features[4] = row['Phones (per 1000)']
    features[5] = row['Infant mortality (per 1000 births)']

    return features


def clusterHelper(clusters, distance_matrix):
    minDist = np.inf
    cluster = None, None
    for i in range(len(clusters)-1):
        for j in range(i+1,len(clusters)):
            #find complete linkage
            distance = distance_matrix[i][j]
            if distance < minDist:
                minDist = distance
                cluster = i, j

    return cluster, minDist




def updateDistMatrix(newCluster, distance_matrix):
    oldCluster1 = int(newCluster[0])
    oldCluster2 = int(newCluster[1])
    #print(distance_matrix)
    newCol = numpy.zeros((len(distance_matrix),1))
    for i in range(len(distance_matrix)):
        newCol[i][0] = max(distance_matrix[oldCluster1][i],distance_matrix[oldCluster2][i])
    distance_matrix = np.hstack((distance_matrix, newCol))

    newRow = np.transpose(newCol)
    newRow = np.append(newCol, 0)
    # print(newRow)
    # print(distance_matrix)
    distance_matrix = np.vstack((distance_matrix, newRow))

    distance_matrix[:, oldCluster1] = np.inf
    distance_matrix[:, oldCluster2] = np.inf
    distance_matrix[oldCluster1, :] = np.inf
    distance_matrix[oldCluster2, :] = np.inf

    return distance_matrix


def hac(features):

    origNumClusters = len(features)
    #print(origNumClusters)
    clusters = {}
    #number clusters
    for number, cluster in enumerate(features):
        clusters[number] = cluster #indexes clusters according to original number
    #print(clusters) - correct
    #distance matrix
    distance_matrix = numpy.zeros((len(clusters),len(clusters)))
    for i in range(len(clusters)-1):
        for j in range(i+1,len(clusters)):
            distance_matrix[i,j] = numpy.linalg.norm(clusters[i]-clusters[j])
            distance_matrix[j, i] = distance_matrix[i, j]


    #print(distance_matrix)

    #(n-1x4 array)
    output = numpy.zeros((len(features)-1,4))
    for i in range(len(output)):
        #print(distance_matrix)
        cluster, minDist = clusterHelper(clusters, distance_matrix)
        pt1,pt2 = cluster
        #find closest clusters and put their numbers into output[i][0] and output[i][1]
        output[i][0] = pt1
        output[i][1] = pt2

        # put complete linkage into output[i][2]
        output[i][2] = minDist

        # put the total number of countries in the cluster into output[i][3]
        # if countries not in clusters add 1 to n, else add the number of countries in the cluster
        n = 0
        #print(pt1, origNumClusters)
        if (pt1<origNumClusters): #original cluster numbers [0,n)
            n+=1
        else:
            n+= clusters[pt1][3]

        if (pt2<origNumClusters):
            n+=1
        else:
            n+= clusters[pt2][3]

        output[i][3] = n
        clusters[origNumClusters+i] = output[i] #merge
        
        distance_matrix = updateDistMatrix(output[i], distance_matrix)
        

    return output


def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    tight_layout()
    return fig


def normalize_features(features):
    for i in range(len(features[0])):  #cols
        col = np.zeros(len(features)) #create empty col to fill in
        for j in range(len(features)):
            col[j] = features[j][i] #fill in col
        mean = numpy.mean(col) #get mean
        sd = numpy.std(col) #get sd
        for j in range(len(features)):
            features[j][i] = (features[j][i]-mean)/sd

    return features


def main():
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 50
    #Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    print(Z_normalized)
    fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()


if __name__ == "__main__":
    main()
