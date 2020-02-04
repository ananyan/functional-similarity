import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
#import netcomp as nc #need to run in conda test_env w/ networkx 1.11 and matplotlib 2.2.3

full_energyharvestertensor = pd.read_excel(r'C:\Users\nandy\Downloads\functionalmodels\functionalmodels\energy_harvesters_tensor.xls', sheet_name=None, header=1, index_col=0) 
energyharvestertensor = pd.concat(full_energyharvestertensor, axis=1)
new = {}
graphs = {}
adj_matrices = {}
for index in energyharvestertensor.index.values:
    new[index] = energyharvestertensor.loc[index].unstack()
    print(index)
    df2 = pd.concat([new[index], new[index].T]).fillna(0)
    graphs[index] = nx.from_numpy_matrix(df2.values)
    adj_matrices[index] = nx.adjacency_matrix(graphs[index])
    mapping = dict(zip(graphs[index], df2.columns.values))
    graphs[index] = nx.relabel_nodes(graphs[index], mapping)

#df = pd.DataFrame()
#for key1 in adj_matrices:
#    for key2 in adj_matrices:
#        A1 = adj_matrices[key1]
#        A2 = adj_matrices[key2]
#        d = nc.lambda_dist(A1,A2,kind='laplacian')
#        df.loc[key1, key2] = d
#        print(key1, key2, d)
#print(df) 
##df.to_csv(r'C:\Users\nandy\Downloads\energy_harvesters_lambdadistance.csv')
plt.subplots(figsize=(20,9))
ax = sns.heatmap(new["Bistable buckling harvester"])
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.figure.tight_layout()
plt.show()
plt.subplots(figsize=(20,9))
ax = sns.heatmap(new[r"Big Belly trash compactor"])
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.figure.tight_layout()
plt.show()
plt.subplots(figsize=(20,9))
ax = sns.heatmap(new["Wing Wave Generator"])
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.figure.tight_layout()
plt.show()
pos = nx.circular_layout(graphs[r'Enocean Eco 100'])
nx.draw(graphs[r'Enocean Eco 100'], pos=pos, with_labels=True, edge_color='red')
plt.show()

##graphedit = pd.DataFrame()
##for key1 in graphs:
##    for key2 in graphs:
##        G1 = graphs[key1]
##        G2 = graphs[key2]
##        A1 = adj_matrices[key1]
##        A2 = adj_matrices[key2]
##        ged = nc.edit_distance(A1,A2)
##        graphedit.loc[key1, key2] = ged
##        print(key1, key2, ged)
##print(graphedit) 
##graphedit.to_csv(r'C:\Users\nandy\Downloads\energy_harvesters_geddistance.csv')

deltacon = pd.DataFrame()
for key1 in graphs:
    for key2 in graphs:
        G1 = graphs[key1]  
        G2 = graphs[key2]
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        dc = nc.deltacon0(A1,A2)
        deltacon.loc[key1, key2] = dc
        print(key1, key2, dc)
print(deltacon) 
#deltacon.to_csv(r'C:\Users\nandy\Downloads\energy_harvesters_deltacon.csv')




#def returnJaccardNetworks(G, G1):
#    #intersection over union 

