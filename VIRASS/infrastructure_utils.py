import geopandas
import numpy as np
from shapely.geometry import Point
import pandas

from . import utils
from . import geo_utils
from . import tree_utils
from . import tree_segmentation

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
            
            
            
@utils.measure_time            
def generate_tree_inventory(points_along_line : pandas.DataFrame, SAT_map, meta_data, radius_meters,
                            tree_mask, tree_species_map = None, nDSM_map = None, mode = "multiscale"):
    
    METERS_to_PIXEL_ratio = meta_data['transform'][0]
    radius = int(radius_meters / METERS_to_PIXEL_ratio) # pixel
    
    print("\n-- Generating tree inventory ")
    crowns = []
    for index in range(0, len(points_along_line)):
        
        # print("Extracting point {} of {}".format(index, len(points_along_line)))
        point = (points_along_line.iloc[index].x, points_along_line.iloc[index].y)
        
        if geo_utils.is_point_valid(point, meta_data, radius):
            # Extract window from satellite image
            W, meta_data_W = geo_utils.extract_window(point, SAT_map, radius, meta_data)
            
            # Extract window from tree_mask.  
            # The tree_mask is pre-generated using the <TreeSegmenter> for computational reasons and provide here as input to the function,
            # instead of being generated inside this loop for every patch.
            # It is still possible to do that but is less efficient.
            tree_mask_W, _ = geo_utils.extract_window(point, tree_mask, radius, meta_data)
            
            # append the crowns extracted at point 'index' into the DataFrame
            if mode == "multiscale":
                crowns.append(tree_utils.extract_tree_crown_multi_scale(geo_utils.multiband_to_RGB(W), 
                                                                        tree_mask_W, 
                                                                        meta_data_W, 
                                                                        verbose = False, 
                                                                        save_output = False,
                                                                        refinement = True,
                                                                        output_filename = "crowns" + str(index) + ".gpkg"))
            else:
                crowns.append(tree_utils.extract_tree_crown(SAT_map = geo_utils.multiband_to_RGB(W), 
                                                                        tree_mask = tree_mask_W, 
                                                                        meta_data = meta_data_W,
                                                                        crown_radius = 4,
                                                                        verbose = False, 
                                                                        save_output = False, 
                                                                        output_filename = "crowns" + str(index) + ".gpkg"))
                    
    crowns = pandas.concat(crowns, copy = False)
    crowns = tree_utils.refine_crowns(crowns, meta_data)
    return crowns
    # crowns.to_file("crowns.gpkg", driver="GPKG") 
    
    
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            