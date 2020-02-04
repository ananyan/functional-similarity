import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt

#Use results of DataProcessing.py
def calcHammingSim(int_datapath):
    data = pd.read_csv(int_datapath,index_col=0, header=2)
    hamming_sim_full = pd.DataFrame(1-squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
    hamming_sim = hamming_sim_full.where(np.triu(np.ones(hamming_sim_full.shape)).astype(np.bool))
    hamming_long = hamming_sim.unstack()
    # Rename columns and turn into a dataframe
    hamming_long.index.rename(['Product A', 'Product B'], inplace=True)
    hamming_long = hamming_long.to_frame('Hamming').reset_index()
    hamming_long = hamming_long[hamming_long['Product A'] != hamming_long['Product B']]
    # Sort the rows of dataframe by column 'hamming'
    hamming_long.dropna(inplace=True)
    hamming_long.reset_index(drop=True, inplace=True)
    hamming_long = hamming_long.sort_values(by ='Hamming' )
    return hamming_long

def calcJaccardSim(int_datapath):
    data = pd.read_csv(int_datapath,index_col=0, header=2)
    jaccard_sim_full = pd.DataFrame(1-squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
    jaccard_sim = jaccard_sim_full.where(np.triu(np.ones(jaccard_sim_full.shape)).astype(np.bool))
    jaccard_long = jaccard_sim.unstack()
    # rename columns and turn into a dataframe
    jaccard_long.index.rename(['Product A', 'Product B'], inplace=True)
    jaccard_long = jaccard_long.to_frame('Jaccard').reset_index()
    jaccard_long = jaccard_long[jaccard_long['Product A'] != jaccard_long['Product B']]
    # sort the rows of dataframe by column 'jaccard'
    jaccard_long.dropna(inplace=True)
    jaccard_long.reset_index(drop=True, inplace=True)
    jaccard_long = jaccard_long.sort_values(by ='Jaccard')
    return jaccard_long

def calcCosineSim(int_datapath):
    data = pd.read_csv(int_datapath,index_col=0, header=2)
    cosine_sim_full = pd.DataFrame(1-squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)
    cosine_sim = cosine_sim_full.where(np.triu(np.ones(cosine_sim_full.shape)).astype(np.bool))
    cosine_long = cosine_sim.unstack()
    # rename columns and turn into a dataframe
    cosine_long.index.rename(['Product A', 'Product B'], inplace=True)
    cosine_long = cosine_long.to_frame('Cosine').reset_index()
    cosine_long = cosine_long[cosine_long['Product A'] != cosine_long['Product B']]
    # sort the rows of dataframe by column 'cosine'
    cosine_long.dropna(inplace=True)
    cosine_long.reset_index(drop=True, inplace=True)
    cosine_long = cosine_long.sort_values(by ='Cosine')
    return cosine_long
    