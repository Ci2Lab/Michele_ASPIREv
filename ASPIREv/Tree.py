import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import random


# Deprecated class, not used anymore
class Tree():
    def __init__(self,):
        
        X = np.random.randint(low = 290968, high = 293157)
        Y = np.random.randint(low = 6805903, high = 6807235)
        
        # Attributes
        self.crown_diameter = np.random.randint(low = 2, high = 20)
        self.geometry = shapely.geometry.Point(X, Y).buffer(self.crown_diameter/2)
        self.height = np.random.rand()
        species = ["spruce", "pine","deciduous"]
        self.species = random.choice(species)
        self.dbh = 0
        self.gap_factor = 0
        self.hegyi_index = 0
        self.risk = 0
        
    
    def calculate_species(self):
        pass
    
    def estimate_DBH(self):
        pass
    
    def calculate_critical_wind_speed(self):
        pass
    
             
    def to_dict(self):
        return {
            'geometry': self.geometry,
            'species' : self.species,
            'height' : self.height,
            'DBH' : self.dbh,
            'gap_factor' : self.gap_factor,
            'hegyi' : self.hegyi_index,
            'risk' : self.risk             
        }
     
        
def generate_forest(N_trees = 200):
    forest = []
    for i in range(0, N_trees):
        tree = Tree()
        forest.append(tree)
    return forest 
 

 
if __name__ == "__main__":      
    forest = generate_forest()
    
    # Put it into a DataFrame
    tree_inventory = pd.DataFrame.from_records([tree.to_dict() for tree in forest])
    tree_inventory = gpd.GeoDataFrame(tree_inventory, crs="EPSG:32632", geometry=tree_inventory['geometry'])
    
    tree_inventory.to_file("tree_inventory.shp")
    
    
    
    