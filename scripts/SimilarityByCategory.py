
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, skew, ttest_1samp, wilcoxon 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns



data = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_vector.csv',index_col=0, header=1)
hamming_sim_full = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccard_sim_full = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosine_sim_full = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)
networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_lambdadistance.csv',index_col=0, header=0)
networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_geddistance.csv',index_col=0, header=0)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_deltacon.csv',index_col=0, header=0)

##Range normalize and turn from distance to similarity
hamming_sim_full_norm = 1 - (hamming_sim_full - hamming_sim_full.min().min())/(hamming_sim_full.max().max() - hamming_sim_full.min().min())
jaccard_sim_full_norm = 1 - (jaccard_sim_full - jaccard_sim_full.min().min())/(jaccard_sim_full.max().max() - jaccard_sim_full.min().min())
cosine_sim_full_norm = 1 - (cosine_sim_full - cosine_sim_full.min().min())/(cosine_sim_full.max().max() - cosine_sim_full.min().min())
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))

##Function to get upper triangle with no diagonal REPEATED FROM RANK CORRELATION.PY
def triu_nodiag(df):
    new = df.where(np.triu(np.ones(df.shape),k=1).astype(np.bool))
    print(new.max().max())
    print(new.max(axis=0).idxmax())
    print(new.max(axis=1).idxmax())
    #print(new.min().min())
    #print(new.min(axis=0).idxmin())
    #print(new.min(axis=1).idxmin())
    new_flat = new.to_numpy().flatten()
    final = new_flat[~np.isnan(new_flat)]
    return final
##Category
Power_tools = [r'De walt sander', 
               r'Real Power Tool Shop', 
               r'Pneumatic Air Ratchet',
               r'SKIL Power Screwdriver-RBS',
               r'B&D Weed Trimmer',
               r'Wagner Paint Roller',
               r'B&D Cordless Screwdriver',
               r'Black & Decker Sander',
               r'B&D Elec. Polisher',
               r'Dremel Electric Engraver-RBS',
               r'Paint Sprayer',
               r'Drill and Nail Extractor  11',
               r'Air Compressor 22',
               r'Power Caulk 20']

Kitchen_tools = [r'Presto Popcorn Popper',
                 r'Salton Sandwich Maker',
                 r'Krups Cheese Grater',
                 r'Dazey Fruit & Veg. Stripper-RBS',
                 r'Presto Salad Shooter',
                 r'Hamilton  Elec. Mixer',
                 r'Takka-1000 Pasta Machine',
                 r'WestBend Food Processor',
                 r'West Bend Mixer',
                 r'Electric Can Opener',
                 r'Regal Ware Electric Knife',
                 r'Black & Decker Elec. Knife',
                 r'KRUPS Café Trio',
                 r'West Bend Iced Tea Maker-RBS',
                 r'Mr. Coffee Iced Tea Maker-RBS',
                 r'Mr. Coffee Coffee Maker-RBS',
                 r'Braun Hand Blender']

Toys = [r'Super Maxx Ball Shooter-RBS',
        r'Toy Solar Car',
        r'Small Projectile Toy',
        r'Tactical Nerf Toy Gun',
        r'OverRider Toy Car',
        r'Horseman Swimming Toy',
        r'Mattel Bubble Extinguisher',
        r'Bumble Ball Toy',
        r'Green Eagle Putting Cup',
        r'Bubble Machine',
        r'F.Price Remote Control Car',
        r'Ertl Bumble Ball',
        r'Fisher Price Bubble Mower',
        r'Good Times Fishing Reel',
        r'Radio Controlled Truck',
        r'Chainsaw Toy Truck 18',
        r'Crossfire Game',
        r'Push-n-Go Train 13']                 
                 
Household = [r'Salon Series 795 Hair Dryer',
             r'Hair Dryer',
             r'Wet/Dry Vacuum',
             r'Stanley Electric Stapler',
             r'Durabuilt Hand Vacuum',
             r'Upright Vacuum Cleaner',
             r'Portable Fan',
             r'Moen Kitchen Water Faucet',
             r'Battery Op. Toothbrush', 
             r'Metronome',	
             r'Humidifier',	
             r'Electric Pencil Sharpener', 
             r'Mini Pro Hair Dryer',
             r'Bissel Hand Vacuum',
             r'Corvette Vac 23',
             r'Tape Rewinder 22',
             r'Hot Glue Gun 21',
             r'Stamp Machine 17']

Vehicle = [r'74 Chevy Tailgate Support-RBS',
           r'Cadillac Slide-Out Visor-RBS',
           r'bicycle brake 18']

category = [Power_tools, Kitchen_tools, Toys, Household, Vehicle]

##Principle
Removers = [r'De walt sander', 
            r'Real Power Tool Shop',
            r'B&D Weed Trimmer',
            r'Black & Decker Sander',
            r'B&D Elec. Polisher',
            r'Dremel Electric Engraver-RBS',
            r'Electric Pencil Sharpener']

AirBlowers = [r'Pneumatic Air Ratchet',
              r'Air Compressor 22',
              r'Salon Series 795 Hair Dryer',
              r'Hair Dryer',
              r'Wet/Dry Vacuum',
              r'Durabuilt Hand Vacuum',
              r'Upright Vacuum Cleaner',
              r'Portable Fan',
              r'Mini Pro Hair Dryer',
              r'Bissel Hand Vacuum',
              r'Corvette Vac 23',]

Shooters = [r'Paint Sprayer',
            r'Power Caulk 20',
            r'Super Maxx Ball Shooter-RBS',
            r'Small Projectile Toy',
            r'Tactical Nerf Toy Gun',
            r'Mattel Bubble Extinguisher',
            r'Bubble Machine',
            r'Hot Glue Gun 21',
            r'Fisher Price Bubble Mower',
            r'Crossfire Game',
            r'Humidifier',
            r'Moen Kitchen Water Faucet']
              
Cookers = [r'Presto Popcorn Popper',
           r'Salton Sandwich Maker']

Mixers = [r'Hamilton  Elec. Mixer',
          r'WestBend Food Processor',
          r'West Bend Mixer',
          r'Braun Hand Blender']
	
Brewers = [r'KRUPS Café Trio',
           r'West Bend Iced Tea Maker-RBS',
           r'Mr. Coffee Iced Tea Maker-RBS',
           r'Mr. Coffee Coffee Maker-RBS']	
	
Vehicles = [r'Toy Solar Car',
            r'OverRider Toy Car',	
            r'F.Price Remote Control Car',
	        r'Radio Controlled Truck',
            r'Chainsaw Toy Truck 18',
            r'Push-n-Go Train 13'] 

Joiners = [r'SKIL Power Screwdriver-RBS',
           r'Stanley Electric Stapler',
           r'B&D Cordless Screwdriver',	
           r'Drill and Nail Extractor  11']
	  
Motion_toys = [r'Horseman Swimming Toy', 
	           r'Bumble Ball Toy',
               r'Ertl Bumble Ball']

Cutters = [r'Electric Can Opener',
           r'Regal Ware Electric Knife',
           r'Black & Decker Elec. Knife',
           r'Presto Salad Shooter',
           r'Krups Cheese Grater',
           r'Dazey Fruit & Veg. Stripper-RBS',
           r'Takka-1000 Pasta Machine']

Winders = [r'Tape Rewinder 22',
           r'Good Times Fishing Reel']

Misc = [r'Green Eagle Putting Cup',
        r'Battery Op. Toothbrush', 
        r'Metronome',	
        r'Stamp Machine 17',
	    r'74 Chevy Tailgate Support-RBS',
        r'Cadillac Slide-Out Visor-RBS',
        r'bicycle brake 18',
        r'Wagner Paint Roller']

principle = [Removers, AirBlowers, Shooters, Cookers, Mixers, Brewers, Vehicles, Joiners, Motion_toys, Cutters, Winders, Misc]

##Another categorization
##Category
Power_tools_all = [r'De walt sander', 
               r'Real Power Tool Shop', 
               r'Pneumatic Air Ratchet',
               r'SKIL Power Screwdriver-RBS',
               r'B&D Weed Trimmer',
               r'Wagner Paint Roller',
               r'B&D Cordless Screwdriver',
               r'Black & Decker Sander',
               r'B&D Elec. Polisher',
               r'Dremel Electric Engraver-RBS',
               r'Paint Sprayer',
               r'Drill and Nail Extractor  11',
               r'Air Compressor 22',
               r'Power Caulk 20']

Kitchen_prep_tools = [r'Presto Popcorn Popper',
                 r'Salton Sandwich Maker',          
                 r'Hamilton  Elec. Mixer',               
                 r'West Bend Mixer',
                 r'KRUPS Café Trio',
                 r'West Bend Iced Tea Maker-RBS',
                 r'Mr. Coffee Iced Tea Maker-RBS',
                 r'Mr. Coffee Coffee Maker-RBS',
                 r'Braun Hand Blender']

kitchen_cutting_tools = [r'Krups Cheese Grater',
                         r'Dazey Fruit & Veg. Stripper-RBS',
                         r'Presto Salad Shooter',
                         r'Takka-1000 Pasta Machine',
                         r'WestBend Food Processor',
                         r'Regal Ware Electric Knife',
                         r'Black & Decker Elec. Knife',
                         r'Electric Can Opener']
            

Shooting_toys = [r'Super Maxx Ball Shooter-RBS',
                 r'Small Projectile Toy',
                 r'Tactical Nerf Toy Gun',
                 r'Mattel Bubble Extinguisher',
                 r'Bumble Ball Toy',
                 r'Bubble Machine',
                 r'Ertl Bumble Ball',
                 r'Fisher Price Bubble Mower',
                 r'Crossfire Game'] 


Car_toys = [r'Toy Solar Car',
            r'OverRider Toy Car',
            r'F.Price Remote Control Car',
            r'Horseman Swimming Toy',
            r'Radio Controlled Truck',
            r'Chainsaw Toy Truck 18',
            r'Push-n-Go Train 13']
                 
Household_items = [r'Salon Series 795 Hair Dryer',
             r'Hair Dryer',
             r'Wet/Dry Vacuum',
             r'Durabuilt Hand Vacuum',
             r'Upright Vacuum Cleaner',
             r'Portable Fan',
             r'Mini Pro Hair Dryer',
             r'Bissel Hand Vacuum',
             r'Corvette Vac 23',
]

Other = [r'74 Chevy Tailgate Support-RBS',
           r'Cadillac Slide-Out Visor-RBS',
           r'bicycle brake 18',
           r'Green Eagle Putting Cup',
           r'Good Times Fishing Reel',
           r'Stanley Electric Stapler',
           r'Moen Kitchen Water Faucet',
           r'Tape Rewinder 22',
           r'Hot Glue Gun 21',
           r'Stamp Machine 17',
           r'Battery Op. Toothbrush', 
           r'Metronome',	
           r'Humidifier',	
           r'Electric Pencil Sharpener']

smaller = [Power_tools_all, Kitchen_prep_tools, kitchen_cutting_tools, Shooting_toys, Car_toys, Household_items, Other]

def categorize_means(categorized, df):	
	#categorized is a list of lists
    #df is the similarity matrix (product names are rows and columns)
    #triu_nodiag is a function that turns the dataframe into an upper triangle without diagonal and flattens it
    categorized = [[ele for ele in sub if ele in list(df.index.values)] for sub in categorized]
    cat_lens = [len(sublist) for sublist in categorized]
    reordered_cols = [val for sublist in categorized for val in sublist]
    print(len(reordered_cols))
    df = df.reindex(reordered_cols)
    df = df[reordered_cols] 
    #find the mean based on the subsection of the dataframe
    start = 0 #start at dataframe index 0
    cat_means = []
    for cat in cat_lens:
        end = start + cat
        cat_means.append(triu_nodiag(df.iloc[start:end,start:end]).mean())
        start = end
    print(cat_lens)
    print(cat_means)
    return cat_means

categorize_means(category, hamming_sim_full_norm)
categorize_means(category, jaccard_sim_full_norm)
categorize_means(category, cosine_sim_full_norm)
categorize_means(category, lambdadist_norm)
categorize_means(category, ged_norm)
categorize_means(category, deltacon_norm)

categorize_means(principle, hamming_sim_full_norm)
categorize_means(principle, jaccard_sim_full_norm)
categorize_means(principle, cosine_sim_full_norm)
categorize_means(principle, lambdadist_norm)
categorize_means(principle, ged_norm)
categorize_means(principle, deltacon_norm)


print(triu_nodiag(hamming_sim_full_norm).mean())
print(triu_nodiag(jaccard_sim_full_norm).mean())
print(triu_nodiag(cosine_sim_full_norm).mean())
print(triu_nodiag(lambdadist_norm).mean())
print(triu_nodiag(ged_norm).mean())
print(triu_nodiag(deltacon_norm).mean())

categorize_means(smaller, hamming_sim_full_norm)
categorize_means(smaller, jaccard_sim_full_norm)
categorize_means(smaller, cosine_sim_full_norm)
categorize_means(smaller, lambdadist_norm)
categorize_means(smaller, ged_norm)
categorize_means(smaller, deltacon_norm)

