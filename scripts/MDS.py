import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

##Load vector form of data
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
hammingdist = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccarddist = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosinedist = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)

##Try a MDS visualization
mds = MDS(n_components = 2, dissimilarity='precomputed', random_state=6)
results = mds.fit(hammingdist)
coords = results.embedding_
plt.subplots_adjust(bottom = 0.1)
plt.scatter(coords[:, 0], coords[:, 1], marker = 'o')
labels = list(data.index)
i=0
for x,y in zip(coords[:,0] ,coords[:,1]):
    label = labels[i]
    i+=1
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') 
plt.show()
