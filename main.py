import VIRASS as ges
import geopandas
import matplotlib.pyplot as plt



if not ("SAT_map" in locals()):
    # Load SAT image
    SAT_map, meta_data = ges.io.open_geoTiFF("_data/WorldView_area.tif")
    
    # Load tree Segmentation mask
    tree_mask = ges.io.open_geoTiFF("_data/trees/tree_segmentation/tree_mask_bn_1_2.tif", with_meta_data = False)   
    
    
    #%% SCAN power line
    
    power_line = ges.io.load_infrastructure_line("_data/power_lines/Kraftnett_files/power_lines.gpkg")
    power_line = ges.infrastructure_utils.clean_line(power_line)
    
    # Convert to a more suitable CRS
    power_line = power_line.to_crs("epsg:32632")
    # power_line.to_file("_data/power_lines/Kraftnett_files/power_lines.gpkg", driver = "GPKG")
        
    # Create corridor
    corridor = ges.infrastructure_utils.create_corridor(power_line, corridor_size = 40)


#%% Tree inventory generation
    

tree_inventory = ges.infrastructure_utils.generate_tree_inventory_along_power_lines(power_line, 
                            satellite_map_file = "_data/WorldView_area.tif", 
                            tree_mask_file = "_data/trees/tree_segmentation/tree_mask_bn_1_2.tif",
                            large_corridor_side_size = 30, small_corridor_side_size = 10,
                            tree_species_map_file = "_data/trees/tree_species/R_refined_2_Q.tif", 
                            nDSM_map_file = "_data/trees/3D_modeling/nDSM_pred.tif", 
                            mode = "multiscale")    


tree_inventory.to_file("_data/trees/crowns/tmp.gpkg", driver = "GPKG")    
# tree_inventory = geopandas.read_file("_data/trees/crowns/tree_inventory")

        

        
            

            
            
























