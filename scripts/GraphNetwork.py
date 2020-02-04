import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#Trying to build a network
def buildFullNetwork(data_longform, metric):
    G = nx.from_pandas_edgelist(data_longform, 'Product A', 'Product B', metric)
    nx.draw(G,pos=nx.spring_layout(G, weight=Metric))
    plt.show()