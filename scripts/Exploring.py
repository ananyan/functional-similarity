import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, skew
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
#data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #data from Weaver method
hamming_sim_full = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccard_sim_full = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosine_sim_full = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)
networkdata = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_lambdadistance.csv',index_col=0, header=0)
networkdata_ged = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_geddistance.csv',index_col=0, header=0)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_deltacon.csv',index_col=0, header=0)

#normalized
hamming_sim_full_norm = 1 - (hamming_sim_full - hamming_sim_full.min().min())/(hamming_sim_full.max().max() - hamming_sim_full.min().min())
jaccard_sim_full_norm = 1 - (jaccard_sim_full - jaccard_sim_full.min().min())/(jaccard_sim_full.max().max() - jaccard_sim_full.min().min())
cosine_sim_full_norm = 1 - (cosine_sim_full - cosine_sim_full.min().min())/(cosine_sim_full.max().max() - cosine_sim_full.min().min())
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))

#upper triangle no diagonal
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

triu_nodiag(hamming_sim_full_norm)
triu_nodiag(jaccard_sim_full_norm)
triu_nodiag(cosine_sim_full_norm)
triu_nodiag(ged_norm)
triu_nodiag(lambdadist_norm)
triu_nodiag(deltacon_norm)


#mean of each domain
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

#domainresults = pd.DataFrame(columns=['inductive','piezoelectric','wind','wave','solar','thermal','hybrid'])
#domainresults = domainresults.append(domainmean(hamming_sim_full_norm), ignore_index=True)
#domainresults = domainresults.append(domainmean(jaccard_sim_full_norm), ignore_index=True)
#domainresults = domainresults.append(domainmean(cosine_sim_full_norm), ignore_index=True)
#domainresults = domainresults.append(domainmean(ged_norm), ignore_index=True)
#domainresults = domainresults.append(domainmean(lambdadist_norm), ignore_index=True)
#domainresults = domainresults.append(domainmean(deltacon_norm), ignore_index=True)
#print(domainresults)
#measures=['hamming', 'jaccard','cosine','ged','spectral','deltacon']


#histograms
def simhistogram(normsim, measurename):
    d1 = normsim.where(np.triu(np.ones(normsim.shape)).astype(np.bool)) #normalized upper triangle
    d1 = d1.to_numpy().flatten()
    d1 = d1[~np.isnan(d1)] #A[~np.isnan(A)]
    plt.hist(d1)
    plt.grid(axis='y')
    plt.title(measurename)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.axvline(d1.mean(), color='k', linewidth=1)
    #plt.axvline(np.median(d1), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(d1.mean(), max_ylim*0.9, 'Mean: {:.2f}'.format(d1.mean()))
    print(skew(d1))
    plt.show()

#simhistogram(hamming_sim_full_norm, 'Hamming')
#simhistogram(jaccard_sim_full_norm, 'Jaccard')
#simhistogram(cosine_sim_full_norm, 'Cosine')
#simhistogram(ged_norm, 'GED')
#simhistogram(lambdadist_norm, 'Spectral')
#simhistogram(deltacon_norm, 'DeltaCon')


#heatmaps of each normalized similarity matrix
def heatmapnorm(normsim):
    plt.subplots(figsize=(20,9))
    ax = sns.heatmap(normsim)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.figure.tight_layout()
    plt.show()

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


allmeasures = pd.DataFrame([hamming_long["Hamming"], jaccard_long["Jaccard"], cosine_long["Cosine"],lambda_long["Lambda"],ged_long["GED"], deltacon_long["DeltaCon"]])
allmeasures_sim = allmeasures.transpose().corr(method='pearson')
def getpearsoncorr(i,j): #use i and j to select the pair of methods to compare
    print(pearsonr(allmeasures.iloc[i,:], allmeasures.iloc[j,:]))
    plt.scatter(allmeasures.iloc[i,:], allmeasures.iloc[j,:])
    plt.show()


## ranks all of the pairs of products
ranked = pd.DataFrame([hamming_rank.index, jaccard_rank.index, cosine_rank.index, lambda_rank.index, ged_rank.index, deltacon_rank.index])
def getspearmancorr(i,j):
    print(spearmanr(ranked.iloc[0,:], ranked.iloc[1,:]))


## comparing the full rankings with different metrics (Hamming, Spearman, Kendall-Tau)
ranked_sim_hamming = pd.DataFrame(1-squareform(pdist(ranked, metric='hamming')), columns = ranked.index, index = ranked.index)
print(ranked_sim_hamming)
ranked_sim_spearman = ranked.transpose().corr(method='spearman')
print(ranked_sim_spearman)
ranked_sim_kendall = ranked.transpose().corr(method='kendall')
print(ranked_sim_kendall)


## do each item ranking separately instead of together
def productrankingcorr(normsim1, normsim2, title):
    corrs = []
    sorted_1 = normsim1#.rank()
    sorted_2 = normsim2#.rank()
    #sorted_1 = normsim1.apply(np.argsort, axis=0) # get indices of each sorted row
    #sorted_2 = normsim2.apply(np.argsort, axis=0)
    for i in range(0,39):
        rankingcorr = sorted_1[sorted_1.columns[i]].corr(sorted_2[sorted_2.columns[i]], method='kendall')
        corrs.append(rankingcorr)
    n = len(corrs)
    reps = 100
    xb = np.random.choice(np.array(corrs), (n,reps)) #bootstrapping
    mb = xb.mean(axis=0)
    confint = np.percentile(mb, [2.5, 97.5])
    #plt.hist(corrs)
    #plt.title(title)
    #plt.show()
    return  (corrs,np.round(confint,2))


(d1,c1) = productrankingcorr(jaccard_sim_full_norm, lambdadist_norm, 'Jaccard-Spectral rank corr')
(d2,c2) = productrankingcorr(hamming_sim_full_norm, jaccard_sim_full_norm, 'Hamming-Jaccard rank corr')
(d3,c3) = productrankingcorr(hamming_sim_full_norm, cosine_sim_full_norm, 'Hamming-Cosine rank corr')
(d4, c4) = productrankingcorr(hamming_sim_full_norm, ged_norm, 'Hamming-GED rank corr')
(d5,c5) = productrankingcorr(hamming_sim_full_norm, lambdadist_norm, 'Hamming-Spectral rank corr')
(d6,c6) = productrankingcorr(hamming_sim_full_norm, deltacon_norm, 'Hamming-DeltaCon rank corr')
(d7,c7) = productrankingcorr(jaccard_sim_full_norm, cosine_sim_full_norm, 'Jaccard-Cosine rank corr')
(d8,c8) = productrankingcorr(jaccard_sim_full_norm, ged_norm, 'Jaccard-GED rank corr')
(d9,c9) = productrankingcorr(jaccard_sim_full_norm, deltacon_norm, 'Jaccard-DeltaCon rank corr')
(d10,c10) = productrankingcorr(cosine_sim_full_norm, ged_norm, 'Cosine-GED rank corr')
(d11,c11) = productrankingcorr(cosine_sim_full_norm, lambdadist_norm, 'Cosine-Spectral rank corr')
(d12,c12) = productrankingcorr(cosine_sim_full_norm, deltacon_norm, 'Cosine-DeltaCon rank corr')
(d13,c13) = productrankingcorr(ged_norm, lambdadist_norm, 'GED-Spectral rank corr')
(d14,c14) = productrankingcorr(ged_norm, deltacon_norm, 'GED-DeltaCon rank corr')
(d15,c15) = productrankingcorr(lambdadist_norm, deltacon_norm, 'Spectral-DeltaCon rank corr')

sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
b = None
f, axes = plt.subplots(5, 5, figsize=(7, 7), sharex=True, sharey=True)
sns.despine()
sns.distplot(d2, kde=False, color="b", ax=axes[0, 0],bins=b)
axes[0,0].set_ylabel('Jaccard')
axes[0,0].title.set_text(c2)
sns.distplot(d3, kde=False, color="b", ax=axes[1, 0],bins=b)
axes[1,0].set_ylabel('Cosine')
axes[1,0].title.set_text(c3)
sns.distplot(d4, kde=False, color="b", ax=axes[2, 0],bins=b)
axes[2,0].set_ylabel('GED')
axes[2,0].title.set_text(c4)
sns.distplot(d5, kde=False, color="b", ax=axes[3, 0], bins=b)
axes[3,0].set_ylabel('Spectral')
axes[3,0].title.set_text(c5)
sns.distplot(d6, kde=False, color="b", ax=axes[4, 0],bins=b)
axes[4,0].set_ylabel('DeltaCon')
axes[4,0].set_xlabel('SMC')
axes[4,0].title.set_text(c6)
sns.distplot(d7, kde=False, color="b", ax=axes[1, 1],bins=b)
axes[1,1].title.set_text(c7)
sns.distplot(d8, kde=False, color="b", ax=axes[2, 1],bins=b)
axes[2,1].title.set_text(c8)
sns.distplot(d1, kde=False, color="b", ax=axes[3, 1],bins=b)
axes[3,1].title.set_text(c1)
sns.distplot(d9, kde=False, color="b", ax=axes[4, 1],bins=b)
axes[4,1].set_xlabel('Jaccard')
axes[4,1].title.set_text(c9)
sns.distplot(d10, kde=False, color="b", ax=axes[2, 2],bins=b)
axes[2,2].title.set_text(c10)
sns.distplot(d11, kde=False, color="b", ax=axes[3, 2],bins=b)
axes[3,2].title.set_text(c11)
sns.distplot(d12, kde=False, color="b", ax=axes[4, 2],bins=b)
axes[4,2].set_xlabel('Cosine')
axes[4,2].title.set_text(c12)
sns.distplot(d13, kde=False, color="b", ax=axes[3, 3],bins=b)
axes[3,3].title.set_text(c13)
sns.distplot(d14, kde=False, color="b", ax=axes[4, 3],bins=b)
axes[4,3].set_xlabel('GED')
axes[4,3].title.set_text(c14)
sns.distplot(d15, kde=False, color="b", ax=axes[4, 4],bins=b)
axes[4,4].title.set_text(c15)
axes[4,4].set_xlabel('Spectral')
for i in range(5):
    for j in range(5):
        if i<j:
            axes[i, j].axis('off')

plt.suptitle('Rank Correlation between Similarity Measures')
#plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


##NETWORK EXPLORATION
#Trying to build a network
G = nx.from_pandas_edgelist(hamming_long, 'Product A', 'Product B', 'Hamming')
nx.draw(G,pos=nx.spring_layout(G, weight='Hamming', seed=1),edge_color='grey')
#plt.show()
# remove connections that are less similar (anything more than 1 std deviation above mean)
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



##Trying to build a network (jaccard)
#G_jaccard = nx.from_pandas_edgelist(jaccard_long, 'Product A', 'Product B', 'Jaccard')
#nx.draw(G_jaccard,pos=nx.spring_layout(G_jaccard, weight='Jaccard'))
#plt.show()
#G2_jaccard = nx.from_numpy_matrix(jaccard_sim_full.values)
#labels = jaccard_sim_full.columns.values
#G2_jaccard = nx.relabel_nodes(G2_jaccard, dict(zip(range(len(labels)), labels)))
#nx.draw(G2_jaccard, pos=nx.spring_layout(G2_jaccard), with_labels=True)
#plt.show()

##Trying to build a network (cosine)
#G_cosine = nx.from_pandas_edgelist(cosine_long, 'Product A', 'Product B', 'Cosine')
#nx.draw(G_cosine,pos=nx.spring_layout(G_cosine, weight='Jaccard'))
#plt.show()
#G2_cosine = nx.from_numpy_matrix(cosine_sim_full.values)
#labels = cosine_sim_full.columns.values
#G2_cosine = nx.relabel_nodes(G2_cosine, dict(zip(range(len(labels)), labels)))
#nx.draw(G2_cosine, pos=nx.spring_layout(G2_cosine), with_labels=True)
#plt.show()

##Set an initial position?

