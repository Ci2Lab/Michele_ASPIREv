import GridEyeS_lib as ges
import numpy as np
import geopandas
import pandas
import matplotlib.pyplot as plt

from skimage import morphology

   

# Load SAT image
SAT_map, meta_data = ges.io.open_geoTiFF("_data/WorldView_area.tif")

# Load tree Segmentation mask
tree_mask = ges.io.open_geoTiFF("_data/trees/tree_segmentation/tree_mask_bn_1_2.tif", with_meta_data = False)   


#%% SCAN power line

power_line = ges.io.load_infrastructure_line("_data/power_lines/Kraftnett_files/power_lines.gpkg")
# Convert to a more suitable CRS
power_line = power_line.to_crs("epsg:32632")

# Scan power line
points_along_line = ges.infrastructure_utils.scan_lines(power_line, distance = 40, save_output = False)


#%% Tree inventory generation

radius = 40
crowns = ges.infrastructure_utils.generate_tree_inventory(points_along_line, SAT_map, meta_data, radius, tree_mask,
                                                          mode = "multiscale")


#%% Add tree species and height

crowns = ges.tree_utils.add_tree_species(crowns, tree_species_map_file = "_data/trees/tree_species/R_refined_1_Q.tif")
crowns = ges.tree_utils.add_tree_height(crowns, nDSM = "_data/trees/3D_modeling/GT/warping/nDSM_warped.tif")

#%% Export it

crowns.to_file("_data/trees/crowns/crowns_246_w_extra_data.gpkg", driver="GPKG") 































