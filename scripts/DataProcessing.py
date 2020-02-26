import pandas as pd
from functools import reduce

##Load raw data (replace with path of energy_harvesters_tensor.xls downloaded in data/01_raw)
full_energyharvestertensor = pd.read_excel(r'C:\Users\nandy\Downloads\functionalmodels\functionalmodels\energy_harvesters_tensor.xls', sheet_name=None, header=1, index_col=0) #set product list as row index, set flows as column names
energyharvestertensor = pd.concat(full_energyharvestertensor, axis=1) #Combined all of the sheets by concatenating horizontally - gets the form for vector measures

##Writes to csv in the form to use for vector measures (replace with path of energy_harvesters_vector.csv as in data/02_intermediate)
energyharvestertensor.to_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv') #Write processed data to csv

##Function to process data to vector
#def processData(rawdata_path, col_labels, row_labels, intdata_path):
#    full_tensor = pd.read_excel(rawdata_path, sheet_name=None, header=col_labels, index_col=row_labels)
#    tensor = pd.concat(full_tensor, axis = 1)
#    tensor.to_csv(intdata_path)

##Get data into format with each product being represented by a vector of which functions and flows it has (not in function-flow pairs)
def logical_merge(A, B):
    # Get all column names
    all_columns = A.columns | B.columns
    # Get column names in common
    common = A.columns & B.columns
    # Get disjoint column names
    _A = [x for x in B.columns if not x in common]
    _B = [x for x in A.columns if not x in common]
    # Logical-or common columns, and concatenate disjoint columns
    return pd.concat([(A | B)[common], A[_B], B[_A]], axis=1)[all_columns]

energyharvesters_flows = reduce(logical_merge, [x[1] for x in full_energyharvestertensor.items()])

for key in full_energyharvestertensor.keys():
    full_energyharvestertensor[key] = full_energyharvestertensor[key].any(axis='columns')
    
energyharvesters_functionsonly = pd.DataFrame.from_dict(full_energyharvestertensor).astype(int)
functionsandflows = energyharvesters_functionsonly.join(energyharvesters_flows, how='left')
print(energyharvesters_functionsonly.any(axis='index')) #Each function is in at least one energy harvester

##Writes to csv in the form to use when you want the form of data used in Weaver 2011 Opportunities for Innovation in Energy Harvesting (replace with path of energy_harvesters_functionsandflows.csv as in data/02_intermediate)
functionsandflows.to_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv') #Write processed data to csv