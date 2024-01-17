import VIRASS as ges
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, label2rgb


plt.close('all')

"""Load Satellite image"""

if not ("SAT_map" in locals()):
    # Load SAT image
    SAT_map, meta_data = ges.io.open_geoTiFF("tree_crown_delineation/image.tif")
    
    # Load tree Segmentation mask
    tree_mask_bn = ges.io.open_geoTiFF("tree_crown_delineation/tree_mask_bn.tif", with_meta_data = False) 
        
    SAT_map = ges.geo_utils.multiband_to_RGB(SAT_map)


SAT_map_small = SAT_map[:200, -200:, :]
tree_mask_bn_small = tree_mask_bn[:200, -200:, :]
plt.style.use('default')

crowns = ges.tree_utils.extract_tree_crown(SAT_map_small, tree_mask_bn_small, crown_element = 4, meta_data = meta_data, scale_analysis = False, 
                                           plot = True, save_output = False, output_filename = "tree_crown_delineation/crown_2.gpkg")


# """Tree crown delineation"""
# crowns = ges.tree_utils.extract_tree_crown_multi_scale(SAT_map, tree_mask_bn, 
#                                                           meta_data = meta_data, refinement = True, 
#                                                           save_output = True,  
#                                                           output_filename = ges.utils.open_dir("Where to save the crowns?") + "final_crowns_refined.gpkg")
    



# # Add tree species    
# import rasterio

# pts = crowns[['pointX', 'pointY']] 
# pts.index = range(len(pts))  
# coords = [(x,y) for x, y in zip(pts.pointX, pts.pointY)]
# src = rasterio.open("tree_crown_delineation/tree_species_map_example.tif")
# # Sample the raster at every point location and store values in DataFrame
# crowns['tree_species'] = [x[0] for x in src.sample(coords)]
# src.close()
# crowns.to_file("crowns_tree_species.gpgk", driver="GPKG")