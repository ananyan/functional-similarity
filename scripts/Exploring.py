import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, skew
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

##Load all data
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
weaverdata = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #data from Weaver method
hamming_sim_full = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccard_sim_full = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosine_sim_full = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)
networkdata = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_lambdadistance.csv',index_col=0, header=0)
networkdata_ged = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_geddistance.csv',index_col=0, header=0)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_deltacon.csv',index_col=0, header=0)

##Range normalize and convert distances to similarity
hamming_sim_full_norm = 1 - (hamming_sim_full - hamming_sim_full.min().min())/(hamming_sim_full.max().max() - hamming_sim_full.min().min())
jaccard_sim_full_norm = 1 - (jaccard_sim_full - jaccard_sim_full.min().min())/(jaccard_sim_full.max().max() - jaccard_sim_full.min().min())
cosine_sim_full_norm = 1 - (cosine_sim_full - cosine_sim_full.min().min())/(cosine_sim_full.max().max() - cosine_sim_full.min().min())
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))

##Getting function diversity for devices
function_diff = pd.DataFrame(abs(np.subtract.outer(np.sum(weaverdata.iloc[:,0:21], axis=1),np.sum(weaverdata.iloc[:,0:21], axis=1))), columns=weaverdata.index, index=weaverdata.index)
print(function_diff)

##Get upper triangle with no diagonal
def triu_nodiag(df):
    new = df.where(np.triu(np.ones(df.shape),k=1).astype(np.bool))
    print(new.max().max())
    print(new.max(axis=0).idxmax())
    print(new.max(axis=1).idxmax())
    print(new.min().min())
    print(new.min(axis=0).idxmin())
    print(new.min(axis=1).idxmin())
    new_flat = new.to_numpy().flatten()
    final = new_flat[~np.isnan(new_flat)]
    return final

##Looking at function diversity and similarity score
plt.scatter(triu_nodiag(function_diff),triu_nodiag(ged_norm))
plt.scatter(triu_nodiag(function_diff),triu_nodiag(lambdadist_norm), marker='^')
plt.scatter(triu_nodiag(function_diff),triu_nodiag(deltacon_norm), marker='s')
plt.scatter(triu_nodiag(function_diff),triu_nodiag(jaccard_sim_full_norm), marker='D')
plt.xlabel('Difference in Number of Functions')
plt.ylabel('Similarity Score')
plt.show()

##Mean of each domain
def domainmean(df):
    inductive = triu_nodiag(df.iloc[0:9, 0:9]).mean()
    piezoelectric = triu_nodiag(df.iloc[9:15, 9:15]).mean()
    wind = triu_nodiag(df.iloc[15:21, 15:21]).mean()
    wave = triu_nodiag(df.iloc[21:24, 21:24]).mean()
    solar = triu_nodiag(df.iloc[24:30, 24:30]).mean()
    thermal = triu_nodiag(df.iloc[30:35, 30:35]).mean()
    hybrid = triu_nodiag(df.iloc[35:39, 0:39]).mean()
    results = {'inductive':inductive,'piezoelectric':piezoelectric,'wind':wind,'wave':wave,'solar':solar,'thermal':thermal,'hybrid':hybrid}
    return results

domainresults = pd.DataFrame(columns=['inductive','piezoelectric','wind','wave','solar','thermal','hybrid'])
domainresults = domainresults.append(domainmean(hamming_sim_full_norm), ignore_index=True)
domainresults = domainresults.append(domainmean(jaccard_sim_full_norm), ignore_index=True)
domainresults = domainresults.append(domainmean(cosine_sim_full_norm), ignore_index=True)
domainresults = domainresults.append(domainmean(ged_norm), ignore_index=True)
domainresults = domainresults.append(domainmean(lambdadist_norm), ignore_index=True)
domainresults = domainresults.append(domainmean(deltacon_norm), ignore_index=True)
print(domainresults)
measures=['hamming', 'jaccard','cosine','ged','spectral','deltacon']

##Heatmaps of each normalized similarity matrix
def heatmapnorm(normsim):
    plt.subplots(figsize=(20,9))
    ax = sns.heatmap(normsim)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.figure.tight_layout()
    plt.show()

##Longform data that shows pairs of products and their similarity score
def getlongform(normsim, measurename):
    measure_sim = normsim.where(np.triu(np.ones(normsim.shape)).astype(np.bool))
    print(measure_sim)
    measure_long = measure_sim.unstack()
    # rename columns and turn into a dataframe
    measure_long.index.rename(['Product A', 'Product B'], inplace=True)
    measure_long = measure_long.to_frame(measurename).reset_index()
    measure_long = measure_long[measure_long['Product A'] != measure_long['Product B']]
    # Sort the rows of dataframe by column 'hamming'
    measure_long.dropna(inplace=True)
    measure_long.reset_index(drop=True, inplace=True)
    measure_rank = measure_long.sort_values(by = measurename )
    return measure_long, measure_rank

hamming_long, hamming_rank = getlongform(hamming_sim_full_norm, 'Hamming')
jaccard_long, jaccard_rank = getlongform(jaccard_sim_full_norm, 'Jaccard')
cosine_long, cosine_rank = getlongform(cosine_sim_full_norm, 'Cosine')
ged_long, ged_rank = getlongform(ged_norm, 'GED')
lambda_long, lambda_rank = getlongform(lambdadist_norm, 'Lambda')
deltacon_long, deltacon_rank = getlongform(deltacon_norm, 'DeltaCon')

##Pearson correlation
allmeasures = pd.DataFrame([hamming_long["Hamming"], jaccard_long["Jaccard"], cosine_long["Cosine"],lambda_long["Lambda"],ged_long["GED"], deltacon_long["DeltaCon"]])
allmeasures_sim = allmeasures.transpose().corr(method='pearson')
def getpearsoncorr(i,j): #use i and j to select the pair of methods to compare
    print(pearsonr(allmeasures.iloc[i,:], allmeasures.iloc[j,:]))
    plt.scatter(allmeasures.iloc[i,:], allmeasures.iloc[j,:])
    plt.show()


##Spearman rank correlation for all of the pairs of products
ranked = pd.DataFrame([hamming_rank.index, jaccard_rank.index, cosine_rank.index, lambda_rank.index, ged_rank.index, deltacon_rank.index])
def getspearmancorr(i,j):
    print(spearmanr(ranked.iloc[0,:], ranked.iloc[1,:]))


##Comparing the full rankings with different metrics (Hamming, Spearman, Kendall-Tau)
ranked_sim_hamming = pd.DataFrame(1-squareform(pdist(ranked, metric='hamming')), columns = ranked.index, index = ranked.index)
print(ranked_sim_hamming)
ranked_sim_spearman = ranked.transpose().corr(method='spearman')
print(ranked_sim_spearman)
ranked_sim_kendall = ranked.transpose().corr(method='kendall')
print(ranked_sim_kendall)


##NETWORK EXPLORATION
##Function to build a network from longform
def buildFullNetwork(data_longform, metric):
    G = nx.from_pandas_edgelist(data_longform, 'Product A', 'Product B', metric)
    nx.draw(G,pos=nx.spring_layout(G, weight=Metric))
    plt.show()

#Trying to build a network
G2 = nx.from_numpy_matrix(hamming_sim_full_norm.values)
labels = hamming_sim_full_norm.columns.values
G2 = nx.relabel_nodes(G2, dict(zip(range(len(labels)), labels)))
nx.draw(G2, pos=nx.spring_layout(G2, seed=1), with_labels=True, edge_color='grey')
#plt.show()

G3 = nx.from_numpy_matrix(jaccard_sim_full_norm.values)
labels = jaccard_sim_full_norm.columns.values
G3 = nx.relabel_nodes(G3, dict(zip(range(len(labels)), labels)))
nx.draw(G3, pos=nx.spring_layout(G3, seed=1), with_labels=True, edge_color='grey')
#plt.show()

G4 = nx.from_numpy_matrix(cosine_sim_full_norm.values)
labels = cosine_sim_full_norm.columns.values
G4 = nx.relabel_nodes(G4, dict(zip(range(len(labels)), labels)))
nx.draw(G4, pos=nx.spring_layout(G4, seed=1), with_labels=True, edge_color='grey')
#plt.show()

#Graphs of all products (hamming)
K = {}
for x in range (0,39):
   K[x]=nx.Graph()
   K[x].add_nodes_from(hamming_sim_full_norm.columns.values)
   for i in range(0,39):
       K[x].add_edge(hamming_sim_full_norm.columns[x], hamming_sim_full_norm.columns[i], weight=hamming_sim_full_norm.iloc[0,i])
plt.figure(2)
nx.draw(K[0], pos=nx.spring_layout(K[0], seed=1), with_labels=True, edge_color='grey')
#plt.show()

#Graphs of all products (jaccard)
K = {}
for x in range (0,39):
   K[x]=nx.Graph()
   K[x].add_nodes_from(jaccard_sim_full_norm.columns.values)
   for i in range(0,39):
       K[x].add_edge(jaccard_sim_full_norm.columns[x], jaccard_sim_full_norm.columns[i], weight=jaccard_sim_full_norm.iloc[0,i])
plt.figure(2)
nx.draw(K[0], pos=nx.spring_layout(K[0], seed=1), with_labels=True, edge_color='grey')
#plt.show()

#Graphs of all products (cosine)
K = {}
for x in range (0,39):
   K[x]=nx.Graph()
   K[x].add_nodes_from(cosine_sim_full_norm.columns.values)
   for i in range(0,39):
       K[x].add_edge(cosine_sim_full_norm.columns[x], cosine_sim_full_norm.columns[i], weight=cosine_sim_full_norm.iloc[0,i])
plt.figure(2)
nx.draw(K[0], pos=nx.spring_layout(K[0], seed=1), with_labels=True, edge_color='grey')
#plt.show()


