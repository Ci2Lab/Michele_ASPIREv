import geopandas
import numpy as np
from shapely.geometry import Point, shape
import rasterio
from rasterio.mask import mask
import pandas

from . import utils
from . import geo_utils
from . import tree_utils
from . import tree_segmentation

from scipy.spatial import distance_matrix


def clean_line(infr_line : geopandas.GeoDataFrame):
    """
    NVE data has some duplicates. Check for them and remove the duplicates
    """
    # Check for duplicates in the lokalID field
    duplicates_mask = infr_line.duplicated(subset = 'lokalID', keep = 'first')
    # duplicates = infr_line[infr_line['lokalID'].duplicated()]
    infr_line = infr_line[~duplicates_mask].reset_index(drop=True)
    return infr_line


def create_corridor(power_line : geopandas.GeoDataFrame, corridor_size = 40):
    # --The styles of caps are: CAP_STYLE.round (1), CAP_STYLE.flat
    # (2), and CAP_STYLE.square (3).
    
    # The styles of joins between offset segments are:
    # JOIN_STYLE.round (1), JOIN_STYLE.mitre (2), and
    # JOIN_STYLE.bevel (3).
    # corridor = power_line.buffer(distance = large_corridor_side_size, cap_style = 3, join_style = 2, mitre_limit = 10)
    
    corridor = power_line.copy()
    corridor['geometry'] = corridor.buffer(distance = corridor_size, cap_style = 2) 
    return power_line


def interpolate_line(line, distance):
    "Interpolate the road geometry. Road is a geometry.linestring.LineString object"
    
    if line.length < distance:
        #No need to interpolate. just pick the centroid
        points = [line.centroid.x, line.centroid.y]
    else:
        #How many points?
        n = int( np.ceil(line.length / distance))
        points = []
        for i in range(0, n):
            tmp = line.interpolate(i*distance)
            points.append( (tmp.x, tmp.y) )
    return np.asarray(points).reshape(-1,2)



def scan_lines(infr_line : geopandas.GeoDataFrame, distance, save_output = False):
    """
    Extract points along the infrastructure line
    """
    locations = []
    for index, line in infr_line.iterrows():
        #get points along road
        locations.append(interpolate_line(line['geometry'], distance))
    
    # locations = list of np.array. Each array contains coordinates along one line.
    # we merge them all into one file since risk will be compute location-wise and not line-wise       
    points = np.concatenate(locations, axis = 0)
    
    # Generate a GeoSeries of points
    points = geopandas.GeoSeries(map(Point, zip(points[:,0], points[:,1])), crs="EPSG:32632")
    if save_output:
        points.to_file(utils.open_dir("Where to save the power line locations?") + "extracted_points.gpkg", driver = 'GPKG')
    return points
            
            
# DEPRECATED            
@utils.measure_time            
def generate_tree_inventory_along_power_lines_old(points_along_line : pandas.DataFrame, SAT_map, meta_data, tree_mask,
                                              radius_meters = 40, tree_species_map_file = None, nDSM_map_file = None, mode = "multiscale"):
    
    tree_inventory = [] 
    # for index in [0]:
    for index, point in points_along_line.items():
        # for each point along the power line  
        point = (points_along_line.iloc[index].x, points_along_line.iloc[index].y)
        
        print(str(index) + "/" + str(len(points_along_line)))
        METERS_to_PIXEL_ratio = meta_data['transform'][0]
        radius = int(radius_meters / METERS_to_PIXEL_ratio) # pixel
        
        W, meta_data_W = geo_utils.extract_window(point, SAT_map, radius, meta_data)
        tree_mask_W, _ = geo_utils.extract_window(point, tree_mask, radius, meta_data)
        
        if W is not None:
            # Extract crowns in the large window W
            crowns = tree_utils.extract_tree_crown_multi_scale(geo_utils.multiband_to_RGB(W), 
                                                                    tree_mask_W, 
                                                                    meta_data_W, 
                                                                    verbose = False, 
                                                                    save_output = False,
                                                                    refinement = True)
            
            
            # ges.tree_utils.plot_crowns(crowns, W, meta_data_W)
            if crowns is not None:
                if not len(crowns.index) == 0:                
                    
                    # Add tree species
                    crowns = tree_utils.add_tree_species(crowns, tree_species_map_file = tree_species_map_file)
                    
                    # Add tree height
                    crowns = tree_utils.add_tree_height(crowns, nDSM = nDSM_map_file)
                    
                    # Extract trees within the corridor
                    distances = distance_matrix(crowns[['pointX', 'pointY']], [point])
                    corridor_side_size = 20
                    trees_within_corridor = crowns[distances < corridor_side_size]
                    
                    if not len(trees_within_corridor.index) == 0:
                        
                        # Estimate CBH from height
                        trees_within_corridor = tree_utils.estimate_DBH(trees_within_corridor)
                        
                        # Finally, calculate the critical wind speed for the trees inside the corridor                       
                        trees_within_corridor = tree_utils.calculate_gap_factor(trees_within_corridor, crowns)
                        # trees_within_corridor = ges.tree_utils.calculate_critical_wind_speed_breakage(trees_within_corridor, 
                        #                                                                               wind_direction = 'E')
                        
                        tree_inventory.append(trees_within_corridor)
                                
                # merged_df = pandas.concat([trees_within_corridor, crowns], axis=0, join='outer')
        
    tree_inventory = pandas.concat(tree_inventory, copy = False, ignore_index=True)
    
 
    
@utils.measure_time          
def generate_tree_inventory_along_power_lines(power_line : pandas.DataFrame, satellite_map_file, tree_mask_file,
                                              large_corridor_side_size = 40,
                                              small_corridor_side_size = 20,
                                              tree_species_map_file = None, nDSM_map_file = None, mode = "multiscale"):

    tree_inventory = [] 
    
    # Make corridor (dilation)    
    corridor = power_line.buffer(distance = 40, cap_style = 2) 
    
    # Open the needed geoTiFF files
    satellite_src = rasterio.open(satellite_map_file)
    satellite_meta_data = satellite_src.profile
    tree_mask_src = rasterio.open(tree_mask_file)
    
    # Copy the metadata from the source raster
    meta_data_corridor = satellite_meta_data.copy()
    
    for index in range(0,10):
    # for index, power_line_segment in corridor.items():
        print(index)
        
        # Clip the satellite image to the corridor       
        SAT_clipped, affine_transform_corridor = mask(satellite_src, [shape(corridor.geometry.values[index])], crop=True)
        SAT_clipped = utils.convert_to_channel_last(SAT_clipped, verbose = False)
        
        # Update metadata with the new dimensions, transform, and CRS
        meta_data_corridor.update({'height': SAT_clipped.shape[0], 
                              'width': SAT_clipped.shape[1], 
                              'transform': affine_transform_corridor, 
                              'dtype': SAT_clipped.dtype})

        # Extract the tree mask 
        tree_mask_clipped, _ = mask(tree_mask_src, [shape(corridor.geometry.values[index])], crop=True)
        tree_mask_clipped = utils.convert_to_channel_last(tree_mask_clipped, verbose = False)
        
        # Extract crowns from the satellite along power line
        crowns = tree_utils.extract_tree_crown_multi_scale(geo_utils.multiband_to_RGB(SAT_clipped), 
                                                                tree_mask_clipped, 
                                                                meta_data_corridor, 
                                                                verbose = False, 
                                                                save_output = False,
                                                                refinement = True)
        
        if crowns is not None:
            if not len(crowns.index) == 0: 
                # ges.tree_utils.plot_crowns(crowns, SAT_clipped, meta_data_corridor)
                
                # Add a field to write the power line segment                 
                crowns['power_line_segment'] = index 
                # Add tree species
                crowns = tree_utils.add_tree_species(crowns, tree_species_map_file = tree_species_map_file)
                
                # Add tree height
                crowns = tree_utils.add_tree_height(crowns, nDSM = nDSM_map_file)
                
                # Calculate distance from the power line
                crowns['dst_to_line'] = crowns['geometry'].apply(lambda point: point.centroid.distance(power_line.geometry.iloc[index]))
                
                # Extract trees within the corridor
                # crowns['within'] = (crowns['dst_to_line'] < small_corridor_side_size).astype(int)                
                
                # Estimate CBH from height for the trees inside the corridor
                trees_within_corridor = crowns[crowns['dst_to_line'] < small_corridor_side_size]
                trees_within_corridor = tree_utils.estimate_DBH(trees_within_corridor)
                    
                # Finally, calculate the critical wind speed for the trees inside the corridor 
                trees_within_corridor = tree_utils.calculate_gap_factor(trees_within_corridor, crowns)
                
                crowns = pandas.merge(crowns, trees_within_corridor, how = 'left')
                
#                     # trees_within_corridor = ges.tree_utils.calculate_critical_wind_speed_breakage(trees_within_corridor, 
#                     #                                                                               wind_direction = 'E')
                    
                tree_inventory.append(crowns)
                        
    tree_inventory = pandas.concat(tree_inventory, copy = False, ignore_index=True)
        
        

    
    
    # Close the geoTiFF files
    satellite_src.close() 
    tree_mask_src.close()           
    
    return tree_inventory
            
            
            
            
            
            
            
            
            
            
            