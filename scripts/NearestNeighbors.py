import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import metrics 
import seaborn as sns

##Load data for vector measures (data/02_intermediate/energy_harvesters_vector.csv or energy_harvesters_functionsandflows.csv)
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
#data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #data from Weaver method

##Calculate hamming, Jaccard, cosine distances
hammingdist = pdist(data, metric='hamming')
jaccarddist = pdist(data, metric='jaccard')
cosinedist = pdist(data, metric='cosine')

##Make sure the distance matrices are labeled (device names)
hamming_dist_full = pd.DataFrame(squareform(hammingdist), columns = data.index, index = data.index)
jaccard_dist_full = pd.DataFrame(squareform(jaccarddist), columns = data.index, index = data.index)
cosine_dist_full = pd.DataFrame(squareform(cosinedist), columns = data.index, index = data.index)


##Load distance matrices for network measures (data/03_processed/energy_harvesters_{...}.csv)
networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_lambdadistance_sorted.csv',index_col=0, header=0)
networkdist = squareform(networkdata) #condensed form
networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_geddistance_sorted.csv',index_col=0, header=0)
networkdist_ged = squareform(networkdata_ged)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_deltacon_sorted.csv',index_col=0, header=0)
networkdist_deltacon = squareform(networkdata_deltacon)

##Range normalize and turn to a measure of similarity (instead of distance)
hamming_dist_full_norm = 1 - ((hamming_dist_full - hamming_dist_full.min().min())/(hamming_dist_full.max().max() - hamming_dist_full.min().min()))
jaccard_dist_full_norm = 1-((jaccard_dist_full - jaccard_dist_full.min().min())/(jaccard_dist_full.max().max() - jaccard_dist_full.min().min()))
cosine_dist_full_norm = 1-((cosine_dist_full - cosine_dist_full.min().min())/(cosine_dist_full.max().max() - cosine_dist_full.min().min()))
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))

##Function for getting nearest neighbors for each device using desired similarity measure and calculating the overlap
def kNNlist(normsim1, normsim2):
    percentoverlap = []
    for systemname in data.index:
        k=6 #set number of neighbors (includes itself)
        x = normsim1.nlargest(k, systemname)[[systemname]].index
        y = normsim2.nlargest(k, systemname)[[systemname]].index
        overlap = len(set(list(x)).intersection(list(y)))-1 #Overlap in the most similar items (not a percent). Excludes itself.
        percentoverlap.append(overlap)
    return percentoverlap

##Function for getting farthest neighbors for each device using desired similarity measure and calculating the overlap [02/25/20 need to fix ...] 
def kNNlistfar(normsim1, normsim2):
    percentoverlap = []
    for systemname in data.index:
        k=5 
        x = normsim1.nsmallest(k, systemname)[[systemname]].index
        y = normsim2.nsmallest(k, systemname)[[systemname]].index
        overlap = len(set(list(x)).intersection(list(y))) #Overlap in the most similar items (not a percent). Excludes itself.
        percentoverlap.append(overlap)
    return percentoverlap

##Look at any device and examine most/least similar devices (use nlargest for most similar systems and nsmallest for least similar systems)
sysname = 'Wing Wave Generator'
#print(jaccard_dist_full_norm.nlargest(38, sysname)[[sysname]])
#print(hamming_dist_full_norm.nlargest(38, sysname)[[sysname]])
#print(cosine_dist_full_norm.nlargest(38, sysname)[[sysname]])
print(ged_norm.nlargest(38, sysname)[[sysname]])
#print(lambdadist_norm.nlargest(38, sysname)[[sysname]])
print(deltacon_norm.nlargest(38, sysname)[[sysname]])

##Getting the overlaps of nearest and farthest devices as a pairwise comparison of different similarity measures
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

e1 = kNNlistfar(jaccard_dist_full_norm, lambdadist_norm)
e2 = kNNlistfar(hamming_dist_full_norm, jaccard_dist_full_norm)
e3 = kNNlistfar(hamming_dist_full_norm, cosine_dist_full_norm)
e4 = kNNlistfar(hamming_dist_full_norm, ged_norm)
e5 = kNNlistfar(hamming_dist_full_norm, lambdadist_norm)
e6 = kNNlistfar(hamming_dist_full_norm, deltacon_norm)
e7 = kNNlistfar(jaccard_dist_full_norm, cosine_dist_full_norm)
e8 = kNNlistfar(jaccard_dist_full_norm, ged_norm)
e9 = kNNlistfar(jaccard_dist_full_norm, deltacon_norm)
e10 = kNNlistfar(cosine_dist_full_norm, ged_norm)
e11 = kNNlistfar(cosine_dist_full_norm, lambdadist_norm)
e12 = kNNlistfar(cosine_dist_full_norm, deltacon_norm)
e13 = kNNlistfar(ged_norm, lambdadist_norm)
e14 = kNNlistfar(ged_norm, deltacon_norm)
e15 = kNNlistfar(lambdadist_norm, deltacon_norm)

##Create pairwise distribution plots
sns.set_palette(sns.dark_palette((260, 75, 60), input="husl"))
sns.set_context("paper")
# Set up the matplotlib figure
b = [0, 1, 2, 3, 4, 5, 6]
f, axes = plt.subplots(5, 5, figsize=(7,7), sharex=True, sharey=True)
width  = 7.5
height = width / 1.618
f.set_size_inches(width,height)
sns.despine()
sns.distplot(d2, kde=False, color="b", ax=axes[0, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e2, kde=False, color="r", ax=axes[0, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[0,0].set_ylabel('Jaccard')
sns.distplot(d3, kde=False, color="b", ax=axes[1, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e3, kde=False, color="r", ax=axes[1, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[1,0].set_ylabel('Cosine')
sns.distplot(d4, kde=False, color="b", ax=axes[2, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e4, kde=False, color="r", ax=axes[2, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[2,0].set_ylabel('GED')
sns.distplot(d5, kde=False, color="b", ax=axes[3, 0], bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e5, kde=False, color="r", ax=axes[3, 0], bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[3,0].set_ylabel('Spectral')
sns.distplot(d6, kde=False, color="b", ax=axes[4, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e6, kde=False, color="r", ax=axes[4, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,0].set_ylabel('DeltaCon')
axes[4,0].set_xlabel('SMC')
sns.distplot(d7, kde=False, color="b", ax=axes[1, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e7, kde=False, color="r", ax=axes[1, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d8, kde=False, color="b", ax=axes[2, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e8, kde=False, color="r", ax=axes[2, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d1, kde=False, color="b", ax=axes[3, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e1, kde=False, color="r", ax=axes[3, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d9, kde=False, color="b", ax=axes[4, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e9, kde=False, color="r", ax=axes[4, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,1].set_xlabel('Jaccard')
sns.distplot(d10, kde=False, color="b", ax=axes[2, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e10, kde=False, color="r", ax=axes[2, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d11, kde=False, color="b", ax=axes[3, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e11, kde=False, color="r", ax=axes[3, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d12, kde=False, color="b", ax=axes[4, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e12, kde=False, color="r", ax=axes[4, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,2].set_xlabel('Cosine')
sns.distplot(d13, kde=False, color="b", ax=axes[3, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e13, kde=False, color="r", ax=axes[3, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(d14, kde=False, color="b", ax=axes[4, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e14, kde=False, color="r", ax=axes[4, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,3].set_xlabel('GED')
sns.distplot(d15, kde=False, color="b", ax=axes[4, 4],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
sns.distplot(e15, kde=False, color="r", ax=axes[4, 4],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,4].set_xlabel('Spectral')
for i in range(5):
    for j in range(5):
       if i<j:
            axes[i, j].axis('off')

#plt.suptitle('Intersection of 5 Nearest and Farthest Products')
#plt.setp(axes, yticks=[])
plt.tight_layout()
plt.subplots_adjust(left=0.1,bottom=0.15)
f.text(0.5,0.02, "Number of Overlapping Systems", ha="center", va="center")
f.text(0.015,0.5, "Frequency", ha="center", va="center", rotation=90)
plt.legend(["Most Similar","Least Similar"],bbox_to_anchor=(1, 6.6), loc='upper right')
plt.setp(axes, xlim=(0,6), xticks=[0,2,4,6], xticklabels=[0,2,4,6])
plt.show()
#f.savefig('RankOverlapV3.pdf', dpi=300)