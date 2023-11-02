import VIRASS as ges
import geopandas
import matplotlib.pyplot as plt


    
    
# --Load power line
power_line = ges.io.load_infrastructure_line("_data/power_lines/Kraftnett_files/power_lines.gpkg", crs = 32632)
power_line = ges.infrastructure_utils.clean_line(power_line)
# power_line.to_file("_data/power_lines/Kraftnett_files/power_lines.gpkg", driver = "GPKG")
    
# Create corridor
# corridor = ges.infrastructure_utils.create_corridor(power_line, corridor_size = 30)

# Tree inventory generation
tree_inventory = ges.infrastructure_utils.generate_tree_inventory_along_power_lines(power_line, 
                            satellite_map_file = "_data/WorldView_area.tif", 
                            tree_mask_file = "_data/trees/tree_segmentation/tree_pred_bn_Unet_attention_segmenter_1_4.tif",
                            large_corridor_side_size = 30, small_corridor_side_size = 10,
                            tree_species_map_file = "_data/trees/tree_species/R_refined_2_Q.tif", 
                            nDSM_map_file = "_data/trees/3D_modeling/nDSM_pred.tif", 
                            mode = "multiscale")    


tree_inventory.to_file("_data/trees/crowns/tree_inventory.gpkg", driver = "GPKG")    
# tree_inventory = geopandas.read_file("_data/trees/crowns/tree_inventory.gpkg")



# Static risk map
static_risk_map = ges.infrastructure_utils.static_risk_map(tree_inventory, power_line_height = 10.8, margin = 2)
static_risk_map.to_file("_data/trees/crowns/static_risk.gpkg", driver = "GPKG")


# Dynamic risk map  
dynamic_risk_map = ges.infrastructure_utils.dynamic_risk_map(tree_inventory, power_line_height = 10.8, margin = 2, 
                                                         wind_direction = 'W',
                                                         wind_gust_speed = 17.4,
                                                         power_line = power_line)
dynamic_risk_map.to_file("_data/trees/crowns/dynamic_risk.gpkg", driver = "GPKG")
 

        

        
            

            
            
























