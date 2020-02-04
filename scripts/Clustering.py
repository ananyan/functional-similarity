import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
import seaborn as sns

data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
#data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #data from Weaver method
hammingdist = pdist(data, metric='hamming')
jaccarddist = pdist(data, metric='jaccard')
cosinedist = pdist(data, metric='cosine')

#labeled
hamming_dist_full = pd.DataFrame(squareform(hammingdist), columns = data.index, index = data.index)
jaccard_dist_full = pd.DataFrame(squareform(jaccarddist), columns = data.index, index = data.index)
cosine_dist_full = pd.DataFrame(squareform(cosinedist), columns = data.index, index = data.index)


#Add network stuff
networkdata = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_lambdadistance.csv',index_col=0, header=0)
networkdist = squareform(networkdata)
networkdata_ged = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_geddistance.csv',index_col=0, header=0)
networkdist_ged = squareform(networkdata_ged)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_deltacon.csv',index_col=0, header=0)
networkdist_deltacon = squareform(networkdata_deltacon)

#normalized
hamming_dist_full_norm = 1 - ((hamming_dist_full - hamming_dist_full.min().min())/(hamming_dist_full.max().max() - hamming_dist_full.min().min()))
jaccard_dist_full_norm = 1-((jaccard_dist_full - jaccard_dist_full.min().min())/(jaccard_dist_full.max().max() - jaccard_dist_full.min().min()))
cosine_dist_full_norm = 1-((cosine_dist_full - cosine_dist_full.min().min())/(cosine_dist_full.max().max() - cosine_dist_full.min().min()))
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))
#Getting nearest neighbors
def kNNlist(normsim1, normsim2):
    percentoverlap = []
    for systemname in data.index:
        k=6
        x = normsim1.nsmallest(k, systemname)[[systemname]].index
        y = normsim2.nsmallest(k, systemname)[[systemname]].index
        overlap = len(set(list(x)).intersection(list(y)))-1 #percent overlap in the most similar items
        percentoverlap.append(overlap)
        
    #plt.hist(percentoverlap, bins=np.arange(-0.5,6.5))
    #plt.show()
    return percentoverlap
print(lambdadist_norm.nlargest(38, 'Wing Wave Generator')[['Wing Wave Generator']])
d1 = kNNlist(jaccard_dist_full_norm, lambdadist_norm)
d2 = kNNlist(hamming_dist_full_norm, jaccard_dist_full_norm)
d3 = kNNlist(hamming_dist_full_norm, cosine_dist_full_norm)
d4 = kNNlist(hamming_dist_full_norm, ged_norm)
d5 = kNNlist(hamming_dist_full_norm, lambdadist_norm)
d6 = kNNlist(hamming_dist_full_norm, deltacon_norm)
d7 = kNNlist(jaccard_dist_full_norm, cosine_dist_full_norm)
d8 = kNNlist(jaccard_dist_full_norm, ged_norm)
d9 = kNNlist(jaccard_dist_full_norm, deltacon_norm)
d10 = kNNlist(cosine_dist_full_norm, ged_norm)
d11 = kNNlist(cosine_dist_full_norm, lambdadist_norm)
d12 = kNNlist(cosine_dist_full_norm, deltacon_norm)
d13 = kNNlist(ged_norm, lambdadist_norm)
d14 = kNNlist(ged_norm, deltacon_norm)
d15 = kNNlist(lambdadist_norm, deltacon_norm)

sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
b = range(0,7,1)
f, axes = plt.subplots(5, 5, figsize=(7,7), sharex=True, sharey=True)
sns.despine()
sns.distplot(d2, kde=False, color="b", ax=axes[0, 0],bins=b)
axes[0,0].set_ylabel('Jaccard')
#axes[0,0].title.set_text(c2)
sns.distplot(d3, kde=False, color="b", ax=axes[1, 0],bins=b)
axes[1,0].set_ylabel('Cosine')
#axes[1,0].title.set_text(c3)
sns.distplot(d4, kde=False, color="b", ax=axes[2, 0],bins=b)
axes[2,0].set_ylabel('GED')
#axes[2,0].title.set_text(c4)
sns.distplot(d5, kde=False, color="b", ax=axes[3, 0], bins=b)
axes[3,0].set_ylabel('Spectral')
#axes[3,0].title.set_text(c5)
sns.distplot(d6, kde=False, color="b", ax=axes[4, 0],bins=b)
axes[4,0].set_ylabel('DeltaCon')
axes[4,0].set_xlabel('SMC')
#axes[4,0].title.set_text(c6)
sns.distplot(d7, kde=False, color="b", ax=axes[1, 1],bins=b)
#axes[1,1].title.set_text(c7)
sns.distplot(d8, kde=False, color="b", ax=axes[2, 1],bins=b)
#axes[2,1].title.set_text(c8)
sns.distplot(d1, kde=False, color="b", ax=axes[3, 1],bins=b)
#axes[3,1].title.set_text(c1)
sns.distplot(d9, kde=False, color="b", ax=axes[4, 1],bins=b)
axes[4,1].set_xlabel('Jaccard')
#axes[4,1].title.set_text(c9)
sns.distplot(d10, kde=False, color="b", ax=axes[2, 2],bins=b)
#axes[2,2].title.set_text(c10)
sns.distplot(d11, kde=False, color="b", ax=axes[3, 2],bins=b)
#axes[3,2].title.set_text(c11)
sns.distplot(d12, kde=False, color="b", ax=axes[4, 2],bins=b)
axes[4,2].set_xlabel('Cosine')
#axes[4,2].title.set_text(c12)
sns.distplot(d13, kde=False, color="b", ax=axes[3, 3],bins=b)
#axes[3,3].title.set_text(c13)
sns.distplot(d14, kde=False, color="b", ax=axes[4, 3],bins=b)
axes[4,3].set_xlabel('GED')
#axes[4,3].title.set_text(c14)
sns.distplot(d15, kde=False, color="b", ax=axes[4, 4],bins=b)
#axes[4,4].title.set_text(c15)
axes[4,4].set_xlabel('Spectral')
for i in range(5):
    for j in range(5):
        if i<j:
            axes[i, j].axis('off')

plt.suptitle('Intersection of 5 Farthest Products')
#plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()



labels = data.index
networklabels = networkdata.index
networklabels_ged = networkdata_ged.index
Z1 = linkage(hammingdist)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z1, labels=labels, leaf_rotation=75)
plt.title('Hamming Distance Clustering')
plt.tight_layout()
plt.show()

cluster_num=7
fl = fcluster(Z1,cluster_num,criterion='maxclust')
clusters = []
clustered_data = {new_list: [] for new_list in range(1,cluster_num+1)} 
for i in range(len(fl)):
    cl = (fl[i],labels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            clustered_data[j].append(item[1])
            print(item[1])

#cluster the data again, but only the largest list length
largest_list = clustered_data[max(clustered_data, key=lambda x:len(clustered_data[x]))]
hammingdist_bigcluster = hamming_dist_full.loc[largest_list, largest_list]
newlabels = hammingdist_bigcluster.index
print(newlabels)
print(squareform(hammingdist_bigcluster))
Z1_2 = linkage(hammingdist_bigcluster)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z1_2, labels=newlabels, leaf_rotation=75)
plt.title('Hamming Distance Clustering 2nd Round')
plt.tight_layout()
plt.show()

cluster_num=6
fl = fcluster(Z1_2,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],newlabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            print(item[1])


#METRIC - JACCARD
Z2 = linkage(jaccarddist)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z2,labels=labels, leaf_rotation=75)
plt.title('Jaccard Distance Clustering')
plt.tight_layout()
plt.show()

cluster_num=7
fl = fcluster(Z2,cluster_num,criterion='maxclust')
clusters = []
clustered_data = {new_list: [] for new_list in range(1,cluster_num+1)} 
for i in range(len(fl)):
    cl = (fl[i],labels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            clustered_data[j].append(item[1])
            print(item[1])

#cluster the data again, but only the largest list length
largest_list = clustered_data[max(clustered_data, key=lambda x:len(clustered_data[x]))]
jaccarddist_bigcluster = jaccard_dist_full.loc[largest_list, largest_list]
newlabels = jaccarddist_bigcluster.index
print(squareform(jaccarddist_bigcluster))
Z2_2 = linkage(jaccarddist_bigcluster)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z2_2, labels=newlabels, leaf_rotation=75)
plt.title('Jaccard Distance Clustering 2nd Round')
plt.tight_layout()
plt.show()

cluster_num=6
fl = fcluster(Z2_2,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],newlabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            print(item[1])

#METRIC - COSINE
Z3 = linkage(cosinedist)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z3, labels=labels, leaf_rotation=75)
plt.title('Cosine Distance Clustering')
plt.tight_layout()
plt.show()

cluster_num=7
fl = fcluster(Z3,cluster_num,criterion='maxclust')
clusters = []
clustered_data = {new_list: [] for new_list in range(1,cluster_num+1)} 
for i in range(len(fl)):
    cl = (fl[i],labels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            clustered_data[j].append(item[1])
            print(item[1])

#cluster the data again, but only the largest list length
largest_list = clustered_data[max(clustered_data, key=lambda x:len(clustered_data[x]))]
cosinedist_bigcluster = cosine_dist_full.loc[largest_list, largest_list]
newlabels = cosinedist_bigcluster.index
print(squareform(cosinedist_bigcluster))
Z3_2 = linkage(cosinedist_bigcluster)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z3_2, labels=newlabels, leaf_rotation=75)
plt.title('Cosine Distance Clustering 2nd Round')
plt.tight_layout()
plt.show()

cluster_num=6
fl = fcluster(Z3_2,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],newlabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            print(item[1])

#Network Lambda Distance
Z4 = linkage(networkdist)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z4, labels=networklabels, leaf_rotation=75)
plt.title('Lambda Distance Clustering')
plt.tight_layout()
plt.show()

cluster_num=7
fl = fcluster(Z4,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],networklabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
clustered_data = {new_list: [] for new_list in range(1,cluster_num+1)} 
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            clustered_data[j].append(item[1])
            print(item[1])

#cluster the data again, but only the largest list length
largest_list = clustered_data[max(clustered_data, key=lambda x:len(clustered_data[x]))]
networkdata_bigcluster = networkdata.loc[largest_list, largest_list]
newlabels = networkdata_bigcluster.index
print(squareform(networkdata_bigcluster))
Z4_2 = linkage(networkdata_bigcluster)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z4_2, labels=newlabels, leaf_rotation=75)
plt.title('Lambda Distance Clustering 2nd Round')
plt.tight_layout()
plt.show()

cluster_num=6
fl = fcluster(Z4_2,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],newlabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            print(item[1])

#Network Graph Edit Distance
Z5 = linkage(networkdist_ged)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z5, labels=networklabels_ged, leaf_rotation=75)
plt.title('GED Clustering')
plt.tight_layout()
plt.show()

cluster_num=7
fl = fcluster(Z5,cluster_num,criterion='maxclust')
clusters = []
for i in range(len(fl)):
    cl = (fl[i],networklabels_ged[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
clustered_data = {new_list: [] for new_list in range(1,cluster_num+1)} 
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            clustered_data[j].append(item[1])
            print(item[1])

#cluster the data again, but only the largest list length
largest_list = clustered_data[max(clustered_data, key=lambda x:len(clustered_data[x]))]
networkdata_ged_bigcluster = networkdata_ged.loc[largest_list, largest_list]
newlabels = networkdata_ged_bigcluster.index
print(squareform(networkdata_ged_bigcluster))
Z5_2 = linkage(networkdata_ged_bigcluster)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z5_2, labels=newlabels, leaf_rotation=75)
plt.title('GED Clustering 2nd Round')
plt.tight_layout()
plt.show()

cluster_num=6
fl = fcluster(Z5_2,cluster_num,criterion='maxclust')
print(fl)
clusters = []
for i in range(len(fl)):
    cl = (fl[i],newlabels[i]) 
    clusters.append(cl)
clusters.sort(key=lambda x:x[0])
for j in range(1,cluster_num+1): #prints which cluster each one is in
    print(j)
    for item in clusters:
        if item[0] == j:
            print(item[1])

#plt.title('Hierarchical Clustering Dendrogram (truncated)')
#plt.xlabel('sample index or (cluster size)')
#plt.ylabel('distance')
#dendrogram(Z1,
#    truncate_mode='lastp',  # show only the last p merged clusters
#    p=5,  # show only the last p merged clusters
#    leaf_rotation=75.,
#    show_contracted=True,  # to get a distribution impression in truncated branches
#)
#plt.show()