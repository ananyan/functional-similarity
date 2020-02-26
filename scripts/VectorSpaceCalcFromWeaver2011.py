import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics

##Testing if the inner product calculation from Weaver 2011 is the same as the cosine similarity
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #Load just numerical information
print(data)

sums = np.sqrt(data.sum(axis=1))
newdata = data.div(sums, axis=0)
final = newdata.dot(newdata.T)
print(final) #same as cosine distance
print(pd.DataFrame(1-squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index))
