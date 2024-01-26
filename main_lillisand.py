import ASPIREv as ges
import numpy as np
import matplotlib.pyplot as plt


# SAT, meta = ges.io.open_geoTiFF("_data_Lillisand/ms.tif")
# SAT = np.take(SAT, indices = [4, 2, 1, 6], axis = -1 )
# ges.io.export_GEOtiff("_data_Lillisand/ms_RGBNIR.tif", SAT, meta)


# Extract alpha layer
# SAT_image, meta_data = ges.io.open_geoTiFF("_data_Lillisand/output_preProcessed.tif")
# alpha = ges.geo_utils.create_alpha_channel(SAT_image, meta_data)
# ges.io.export_GEOtiff("_data_Lillisand/alpha_channel.tif", alpha, meta_data)
# ges.geo_utils.raster_to_vector(input_raster = "_data_Lillisand/alpha_channel.tif", 
#                                output_vector = "_data_Lillisand/alpha_channel.gpkg")


# # --Load power line
power_line = ges.io.load_infrastructure_line("_data_Lillisand/power_lines_clipped.gpkg", crs = 32632)

# # Create corridor
# corridor = ges.infrastructure_utils.create_corridor(power_line, corridor_size = 30)
# corridor.to_file("_data_Lillisand/corridor_30m.gpkg", driver = "GPKG")   

tree_inventory = ges.infrastructure_utils.generate_tree_inventory_along_power_lines(power_line, 
                            satellite_map_file = "_data_Lillisand/output_preProcessed.tif", 
                            tree_mask_file = "_data_Lillisand/Unet_attention_segmenter_RGBNIR_bn.tif",
                            large_corridor_side_size = 30, small_corridor_side_size = 10,
                            mode = "multiscale") 

tree_inventory.to_file("_data_Lillisand/tree_inventory.gpkg", driver = "GPKG")  





