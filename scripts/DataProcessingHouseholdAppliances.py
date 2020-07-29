import pandas as pd
import numpy as np

full_appliancesvector = pd.read_excel(r'C:\Users\nandy\Downloads\householdappliances_vector.xls', sheet_name=r'Phi matrix-Fig. D.1', header=0, index_col=0) #set functions as row index, set products as column names
full_appliancesvector = full_appliancesvector[:-5]
full_appliancesvector = full_appliancesvector[full_appliancesvector.columns[:-2]]
full_appliancesvector.where(full_appliancesvector == 0, other=1, inplace=True)
products = list(full_appliancesvector.columns)
full_appliancesvector = full_appliancesvector.reset_index()

##Simple cleaning
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'allow DOF of', r'allowDOFof', regex=False)
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'solids', r'solid', regex=False)
#full_appliancesvector['Function'] = full_appliancesvector[r'Sub-function  \  Device'].str.split(' ').str[0]
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human en.', r'human energy', regex=False)
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human force', r'human energy', regex=False)
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human hand', r'human', regex=False)
#full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'control signal', r'control', regex=False)
#full_appliancesvector['Flow'] = full_appliancesvector[r'Sub-function  \  Device'].str.split(' to ').str[0]
#full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.split(' and ').str[0]
#full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.split(' ',1).str[1]
#full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.strip()

##More complex cleaning
##Remove functions that operate on the device/product level
searchfor = ['product', 'device','corrosion','store energy','store mechanical energy','supply mechanical energy','transmit energy','stop chemical energy','export sound','dissipate sound','change sound','mix liquid and gas','mix liquid and solid'] #last 4 added for V2 #added last 6 for V3
full_appliancesvector = full_appliancesvector[~full_appliancesvector[r'Sub-function  \  Device'].str.contains('|'.join(searchfor))]
full_appliancesvector.drop([r'Chainsaw Toy Truck 18', r'Push-n-Go Train 13', r'Good Times Fishing Reel', r'Metronome', r'Stanley Electric Stapler', 
                            r'Small Projectile Toy', r'Tactical Nerf Toy Gun', r'Drill and Nail Extractor  11', r'KRUPS Café Trio'], axis=1, inplace=True) #added for V2
full_appliancesvector.drop([r'Wet/Dry Vacuum',r'Durabuilt Hand Vacuum',r'Horseman Swimming Toy', r'Mattel Bubble Extinguisher', r'Electric Can Opener',
                            r'Bubble Machine',r'Humidifier',r'West Bend Iced Tea Maker-RBS',r'Mr. Coffee Iced Tea Maker-RBS',r'Mr. Coffee Coffee Maker-RBS',
                            r'Radio Controlled Truck',r'De walt sander',r'Krups Cheese Grater',r'Salon Series 795 Hair Dryer',r'Presto Salad Shooter',r'Toy Solar Car',
                            r'Hair Dryer',r'Hamilton  Elec. Mixer',r'OverRider Toy Car',r'Regal Ware Electric Knife',r'Takka-1000 Pasta Machine',r'Upright Vacuum Cleaner',
                            r'WestBend Food Processor',r'Black & Decker Sander',r'West Bend Mixer',r'F.Price Remote Control Car',r'Portable Fan',r'B&D Elec. Polisher',
                            r'Black & Decker Elec. Knife',r'Battery Op. Toothbrush',r'Electric Pencil Sharpener',r'Mini Pro Hair Dryer', r'Braun Hand Blender',
                            r'Air Compressor 22', r'Corvette Vac 23',r'Tape Rewinder 22',r'Ertl Bumble Ball', r'Fisher Price Bubble Mower'], axis=1, inplace=True) #added for V3

##Changing signal specification
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'import signal', r'import control', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'display signal', r'display status', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'transmit signal', r'transmit control', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'separate signal', r'separate control', regex=False) #added for V2

##Change function specification from tertiary to secondary
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'allow DOF', r'allowDOF', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(allowDOF|translate|rotate)', r'guide', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(divide|extract|remove|refine)', r'separate', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(transport|transmit)', r'transfer', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(join|link|connect)', r'couple', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'^assemble', r'couple', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(increase|decrease|maintain)', r'regulate', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(increment|decrement|shape|condition|form)', r'change', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(prevent|inhibit|resist)', r'stop', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(contain|collect)', r'store', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(track|display)', r'indicate', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(dissipate)', r'distribute', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(detect|measure)', r'sense', regex=True)
full_appliancesvector['Function'] = full_appliancesvector[r'Sub-function  \  Device'].str.split(' ').str[0]

#Match to energy harvester format (added for V2)
full_appliancesvector['Function'] = full_appliancesvector['Function'].str.title()
print(full_appliancesvector['Function'].unique()) ##Missing "process" function

##Change flow specifications
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'of', r'', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'solids', r'solid', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'vibrations', r'vibration', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human en.', r'Human E', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human force', r'Human E', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'human hand', r'Human', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'electricity', r'EE', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'sound', r'acoustic energy', regex=False) #changed "sound" --> "acoustic energy" instead of "status" for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'(heat|temperature)', r'therm E', regex=True)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'sound', r'status', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'displacement', r'translation', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'control signal', r'control', regex=False)
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'solar energy', r'light E', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'optical energy', r'light E', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'liquid and gas', r'mixture', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'liquid and solid', r'mixture', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'pressure', r'hydraulics', regex=False) #added for V2
full_appliancesvector[r'Sub-function  \  Device'] = full_appliancesvector[r'Sub-function  \  Device'].str.replace(r'friction', r'therm E', regex=False) #added for V2


full_appliancesvector['Flow'] = full_appliancesvector[r'Sub-function  \  Device'].str.split(' to ').str[0]
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.split(' and ').str[0]
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.split(' ',1).str[1]
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.strip()
full_appliancesvector.drop(columns=[r'Sub-function  \  Device'],inplace=True)

#Match to energy harvester format (added for V2)
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'solid', r'Solid', regex=False)
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'gas', r'Gas', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'liquid', r'Liquid', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'pneumatics', r'pneu E', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'hydraulics', r'hyd E', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'rotation', r'rot ME', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'translation', r'trans ME', regex=False) 
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'vibration', r'vib ME', regex=False)
full_appliancesvector['Flow'] = full_appliancesvector['Flow'].str.replace(r'human energy', r'Human E', regex=False) 
print(full_appliancesvector['Flow'].unique())
print(full_appliancesvector)

##Format for vectors
appliancesvector = full_appliancesvector.groupby(['Function','Flow']).sum()
appliancesvector.where(appliancesvector == 0, other=1, inplace=True)
appliancesvector = appliancesvector.T
appliancesvector.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_vector_V3.csv') #Write processed data to csv


##Format for networks
appliancesnetwork = full_appliancesvector.groupby(['Function','Flow']).sum().unstack().fillna(0)
appliancesnetwork.where(appliancesnetwork == 0, other=1, inplace=True)
appliancesnetwork.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_adjacencymatrix_V3.csv') #Write processed data to csv
