import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
#import netcomp as nc #need to run in conda test_env w/ networkx 1.11 and matplotlib 2.2.3

##Load original data
#full_energyharvestertensor = pd.read_excel('../data/01_raw/energy_harvesters_tensor.xls', sheet_name=None, header=1, index_col=0) 
#energyharvestertensor = pd.concat(full_energyharvestertensor, axis=1)

##Load appliance/toy data
appliancestensor = pd.read_csv('../data/02_intermediate/appliances_and_toys_adjacencymatrix_V2.csv', header=[0,1], index_col=0) 
appliancestensor = appliancestensor.stack(0)
appliancestensor = appliancestensor.unstack(0)
energyharvestertensor = appliancestensor #temporarily rename just to run because I didn't use functions ... :(

##Load combined data
#combinedtensor = pd.read_csv('../data/02_intermediate/combined_adjacencymatrix.csv', header=[0,1], index_col=0) 
#combinedtensor = pd.read_csv('../data/02_intermediate/limitedcombined_adjacencymatrix.csv', header=[0,1], index_col=0) 
#energyharvestertensor = combinedtensor #temporarily rename just to run because I didn't use functions ... :( 

##Map each device as a network using NetworkX
new = {}
graphs = {}
adj_matrices = {}
for index in energyharvestertensor.index.values:
    new[index] = energyharvestertensor.loc[index].unstack()
    print(index)
    df2 = pd.concat([new[index], new[index].T], sort=True).fillna(0).sort_index() #ADDED a sort - energy harvesters rerun using tag "sorted" ... :(
    graphs[index] = nx.from_numpy_matrix(df2.values)
    adj_matrices[index] = nx.adjacency_matrix(graphs[index])
    mapping = dict(zip(graphs[index], df2.columns.values)) 
    graphs[index] = nx.relabel_nodes(graphs[index], mapping)


def maxofdict(d):
    mx_pair = max(zip(d.values(), d.keys()))
    return mx_pair

##Calculate properties of the graphs
properties = pd.DataFrame()
for key in graphs:
    print(graphs[key])
    deg_centrality = nx.algorithms.centrality.degree_centrality(graphs[key])
    cent, nname = maxofdict(deg_centrality)
    properties.loc[key, "max_deg_centrality"] = cent
    properties.loc[key, "avg_deg_centrality"] = sum(deg_centrality.values())/len(deg_centrality)
    max_degree = nx.adjacency_matrix(graphs[key]).sum(axis=1).max()
    properties.loc[key, "max_degree"] = max_degree
    max_deg_nodes = [item[0] for item in graphs[key].degree() if item[1] == max_degree]
    max_deg_nodes_str = ','.join(max_deg_nodes)
    properties.loc[key, "max_deg_nodes"] = max_deg_nodes_str
    complexity = adj_matrices[key].sum().sum()/2
    properties.loc[key, "complexity"] = complexity
print(properties)
#properties.to_csv('../data/03_processed/energy_harvesters_properties.csv')
#properties.to_csv('../data/03_processed/appliances_and_toys_V2_properties.csv')


##Calculate the Laplacian Spectral Distance (pnorm 2) using NetComp 
df = pd.DataFrame()
for key1 in adj_matrices:
    for key2 in adj_matrices:
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        d = nc.lambda_dist(A1,A2,kind='laplacian')
        df.loc[key1, key2] = d
        print(key1, key2, d)
print(df) 

#df.to_csv('../data/03_processed/energy_harvesters_lambdadistance.csv')
#df.to_csv('../data/03_processed/appliances_and_toys_lambdadistance_V2.csv')
#df.to_csv('../data/03_processed/combined_lambdadistance.csv')
#df.to_csv('../data/03_processed/limitedcombined_lambdadistance.csv')


##Calculate the Graph Edit Distance using NetComp
graphedit = pd.DataFrame()
for key1 in graphs:
    for key2 in graphs:
        G1 = graphs[key1]
        G2 = graphs[key2]
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        ged = nc.edit_distance(A1,A2)
        graphedit.loc[key1, key2] = ged
        print(key1, key2, ged)
print(graphedit) 
#graphedit.to_csv('../data/03_processed/energy_harvesters_geddistance.csv')
#graphedit.to_csv('../data/03_processed/appliances_and_toys_geddistance_V2.csv')
#graphedit.to_csv('../data/03_processed/combined_geddistance.csv')
#graphedit.to_csv('../data/03_processed/limitedcombined_geddistance.csv')


##Calculate the DeltaCon Distance using NetComp
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
#deltacon.to_csv('../data/03_processed/energy_harvesters_deltacon.csv')
#deltacon.to_csv('../data/03_processed/appliances_and_toys_deltacon_V2.csv')
#deltacon.to_csv('../data/03_processed/combined_deltacon.csv')
#deltacon.to_csv('../data/03_processed/limitedcombined_deltacon.csv')

##Calculate the Vertex Edge Overlap Distance using NetComp
ve_overlap = pd.DataFrame()
for key1 in graphs:
    for key2 in graphs:
        G1 = graphs[key1]  
        G2 = graphs[key2]
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        veo = nc.vertex_edge_distance(A1,A2)
        ve_overlap.loc[key1, key2] = veo
        print(key1, key2, veo)
print(ve_overlap)
#ve_overlap.to_csv('../data/03_processed/energy_harvesters_vertexedgeoverlap.csv')
#ve_overlap.to_csv('../data/03_processed/appliances_and_toys_vertexedgeoverlap_V2.csv')



##Calculate the NetSimile Distance using NetComp
netsimile = pd.DataFrame()
for key1 in graphs:
    for key2 in graphs:
        G1 = graphs[key1]  
        G2 = graphs[key2]
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        ns = nc.netsimile(A1,A2)
        netsimile.loc[key1, key2] = ns
        print(key1, key2, ns)
print(netsimile)
#netsimile.to_csv('../data/03_processed/energy_harvesters_netsimile.csv')
#netsimile.to_csv('../data/03_processed/appliances_and_toys_netsimile_V2.csv')


##Calculate the Resistance Distance using NetComp
resistance = pd.DataFrame()
for key1 in graphs:
    for key2 in graphs:
        G1 = graphs[key1]  
        G2 = graphs[key2]
        A1 = adj_matrices[key1]
        A2 = adj_matrices[key2]
        rd = nc.resistance_distance(A1,A2, renormalized=True)
        resistance.loc[key1, key2] = rd
        print(key1, key2, rd)
print(resistance)
#resistance.to_csv('../data/03_processed/energy_harvesters_resistance.csv')
#resistance.to_csv('../data/03_processed/appliances_and_toys_resistance_V2.csv')





