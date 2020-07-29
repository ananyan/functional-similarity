import pandas as pd
import numpy as np

##Load original data
full_energyharvestertensor = pd.read_excel(r'C:\Users\nandy\Downloads\functionalmodels\functionalmodels\energy_harvesters_tensor.xls', sheet_name=None, header=1, index_col=0) 
energyharvestertensor = pd.concat(full_energyharvestertensor, axis=1)
energyharvestertensor.sort_index(axis=1, level=0, inplace=True) #sort everything


##Load appliance/toy data
appliancestensor = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_adjacencymatrix_V2.csv', header=[0,1], index_col=0) 
appliancestensor = appliancestensor.stack(0)
appliancestensor = appliancestensor.unstack(0)
appliancestensor.columns = appliancestensor.columns.swaplevel(0, 1) #swap the indices
appliancestensor.sort_index(axis=1, level=0, inplace=True) #sort everything

##Combine
combined_adj = pd.concat([energyharvestertensor, appliancestensor], sort=False).fillna(0)
#combined_adj.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\combined_adjacencymatrix.csv') #Write processed data to csv (energy harvester + appliances and toys V2)

combined_adj['ex1'] = 0
combined_adj['ex2'] = 0
combined_adj['ex3'] = 0
#combined_adj.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\combined_vector.csv') 

##Combine with V3 and remove things from energy harvesting dataset that don't match
limited_eh = energyharvestertensor.iloc[[11,12,13,14,15,22,26,28,30,31,32,34,35,36],:]
limited_eh.drop('mag E', axis=1, level=1, inplace=True)
limited_eh.drop('Process', axis=1, level=0, inplace=True)

limitedappliancestensor = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_adjacencymatrix_V3.csv', header=[0,1], index_col=0) 
limitedappliancestensor = limitedappliancestensor.stack(0)
limitedappliancestensor = limitedappliancestensor.unstack(0)
limitedappliancestensor.columns = limitedappliancestensor.columns.swaplevel(0, 1) #swap the indices
limitedappliancestensor.sort_index(axis=1, level=0, inplace=True) #sort everything
limited_combined = pd.concat([limited_eh, limitedappliancestensor], sort=False)
limited_combined.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\limitedcombined_adjacencymatrix.csv') 
