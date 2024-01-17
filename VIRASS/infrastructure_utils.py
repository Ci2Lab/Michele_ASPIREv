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
    return corridor


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



@utils.measure_time 
def scan_lines(infr_line : geopandas.GeoDataFrame, distance, save_output = False):
    """
    Extract points along the infrastructure line
    """
    
    # Initialize an empty list to store the points
    locations = []
    
    # Iterate through the infrastructure lines
    for geometry in infr_line['geometry']:
        # Create points along the line
        points = interpolate_line(geometry, distance)
        locations.append(points)
        
    # Concatenate the arrays into a single array
    points = np.concatenate(locations, axis=0)  
    
    # Convert the points array into a GeoSeries
    geo_points = geopandas.GeoSeries([Point(x, y) for x, y in points], crs="EPSG:32632")
    
    if save_output:
        # Save the GeoSeries to a GeoPackage file
        output_path = "extracted_points.gpkg"
        geo_points.to_file(output_path, driver='GPKG')
        
    return geo_points

            
            
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
    corridor = power_line.buffer(distance = large_corridor_side_size, cap_style = 2) 
    
    # Open the needed geoTiFF files
    satellite_src = rasterio.open(satellite_map_file)
    satellite_meta_data = satellite_src.profile
    tree_mask_src = rasterio.open(tree_mask_file)
    
    # Copy the metadata from the source raster
    meta_data_corridor = satellite_meta_data.copy()
    
    # for index in range(0,15):
    for index, power_line_segment in corridor.items():
        print(str(index) + "/" + str(len(corridor.index)))
        
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
                # crowns = tree_utils.add_tree_species(crowns, tree_species_map_file = tree_species_map_file)
                
                # Add tree height
                # crowns = tree_utils.add_tree_height(crowns, nDSM = nDSM_map_file)
                
                # Calculate distance from the power line
                crowns['dst_to_line'] = crowns['geometry'].apply(lambda point: point.centroid.distance(power_line.geometry.iloc[index]))
                
                # Extract trees within the corridor
                # crowns['within'] = (crowns['dst_to_line'] < small_corridor_side_size).astype(int)                
                
                # Estimate CBH from height for the trees inside the corridor
                # trees_within_corridor = crowns[crowns['dst_to_line'] < small_corridor_side_size]
                # trees_within_corridor = tree_utils.estimate_DBH(trees_within_corridor)
                    
                # Finally, calculate the critical wind speed for the trees inside the corridor 
                # trees_within_corridor = tree_utils.calculate_shield_factor(trees_within_corridor, crowns)                
                # crowns = pandas.merge(crowns, trees_within_corridor, how = 'left')
                
#                     # trees_within_corridor = ges.tree_utils.calculate_critical_wind_speed_breakage(trees_within_corridor, 
#                     #                                                                               wind_direction = 'E')
                    
                tree_inventory.append(crowns)
                        
    tree_inventory = pandas.concat(tree_inventory, copy = False, ignore_index=True)
        
        

    
    
    # Close the geoTiFF files
    satellite_src.close() 
    tree_mask_src.close()           
    
    return tree_inventory



@utils.measure_time
def static_risk_map(tree_inventory, small_corridor_side_size = 10, power_line_height = 10.8, margin = 1.5):
    # Create a copy of the relevant columns from the tree_inventory DataFrame
    static_risk_map = tree_inventory[["pointX", "pointY", "geometry", "dst_to_line", "tree_height", "power_line_segment"]].copy()
    
    # Calculate the threat based on distance and height
    threat = np.sqrt(static_risk_map['dst_to_line']**2 + power_line_height**2) - margin - static_risk_map['tree_height']
    
    # Assign static risk values
    static_risk_map['can_hit'] = np.where(threat <= 0, 1, 0)
    
    # Set trees outside the corridor a static risk of 0
    static_risk_map['can_hit'] = np.where(static_risk_map['dst_to_line'] > small_corridor_side_size, 0, static_risk_map['can_hit'])
    
    return static_risk_map



@utils.measure_time
def dynamic_risk_map(tree_inventory, wind_direction = 'E', wind_gust_speed = 20,  power_line_height = 10.8, margin = 1.5, power_line = None ):
    dynamic_risk_map = tree_inventory.copy()
    
    # Calculate the critical wind speed
    dynamic_risk_map = tree_utils.calculate_critical_wind_speed_breakage(dynamic_risk_map, wind_direction = wind_direction)
    assert 'Crit_wspeed_breakage' in dynamic_risk_map.columns
    
    # Assign dynamic risk values
    dynamic_risk_map['can_fall'] = np.where(dynamic_risk_map['Crit_wspeed_breakage'] >= wind_gust_speed, 0, 1)
    
    
    def _angle_from_wind_rose_category(category):
        categories = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        if category in categories:
            index = categories.index(category)
            return (index * 45) % 360
        else:
            raise ValueError("Invalid wind rose category")
            
            
    def calculate_can_hit(row):
        if power_line is not None:
            power_line_segment = power_line.iloc[row['power_line_segment']]['geometry']
            nearest_point = power_line_segment.interpolate(power_line_segment.project(Point(row['pointX'], row['pointY'])))
            angle = (np.degrees(np.arctan2(row['pointY'] - nearest_point.y, row['pointX'] - nearest_point.x)) + 180) % 360
            return int(abs(angle - _angle_from_wind_rose_category(wind_direction)) <= 22.5)
        return 0
    
    
    is_facing_power_line = dynamic_risk_map.apply(calculate_can_hit, axis=1)
    threat = np.sqrt(dynamic_risk_map['dst_to_line']**2 + power_line_height**2) - margin - dynamic_risk_map['tree_height']
    can_collide = np.where(threat <= 0, 1, 0)
    
    # Trees can hit the power line if they are facing the power line and can collide with it
    dynamic_risk_map['can_hit'] = is_facing_power_line * can_collide


    # only trees that can fall can hit
    dynamic_risk_map['can_hit'] = dynamic_risk_map['can_hit'] * dynamic_risk_map['can_fall'] 
    return dynamic_risk_map

    

def assign_risk_to_power_line(power_line, risk_map, point_distance = 20, trees_nearby = 10, 
                              save_output = False, save_path = ""):
    """
    Compute points along the power line and count how many dangerous trees there are nearby 
    """
    # Extract points along the power line every 'point_distance'
    points_along_line = scan_lines(power_line, distance = point_distance)
    assert 'can_hit' in risk_map
    dangerous_trees = risk_map[risk_map['can_hit'] == 1]['geometry'].centroid 

    # Initialize an empty list to store counts
    counts = []
    for point in points_along_line:
        # Calculate the number of dangerous trees within 10 meters of the point
        count = sum(point.distance(dangerous_tree) < trees_nearby for dangerous_tree in dangerous_trees)
        counts.append(count)

    # Convert the list of counts to a NumPy array
    counts_array = np.array(counts)   

    geo_df = geopandas.GeoDataFrame({'geometry': points_along_line})
    geo_df['counts'] = counts_array
    
    if save_output:
        geo_df.to_file(save_path, driver = "GPKG")
    return geo_df
    

            
            
            
            
            
            
            
            
            
            
            