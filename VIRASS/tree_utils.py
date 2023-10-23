import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology
from skimage.color import rgb2gray, label2rgb
from skimage.filters  import gaussian
from skimage.util import invert
from skimage.segmentation import watershed, mark_boundaries, find_boundaries
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import geopandas as gpd
import pandas
from shapely.geometry import Point

from scipy.spatial import distance_matrix
import rasterio

from . import geo_utils

# To avoid warning: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pandas.options.mode.chained_assignment = None  # default='warn'

def extract_tree_crown(SAT_map, tree_mask, crown_element = 4, meta_data = None, 
                       scale_analysis = False, plot = False, verbose = True,
                       save_output = True, 
                       output_filename = "tree_crown.gpkg"):
    """ 
    Extract the tree crowns (centroid locations and radius) from an image.
    The approach is based on:
        Jing, L.; Hu, B.; Noland, T.; Li, J. 
        An Individual Tree Crown Delineation Method Based on Multi-Scale Segmentation of Imagery. 
        ISPRS J. Photogramm. Remote Sens. 2012, 70, 88–98. 
        https://www.sciencedirect.com/science/article/pii/S0924271612000767
    
    Parameters
    ----------
    SAT_map: satellite image
    tree_mask: output of the tree segmenter. Binary mask for trees-no trees
    crown_element: structuring element to use (measured in pixel). 
    meta_data: geoTiFF meta data of the SAT_map
    scale_analysis: BOOL, whether to automatically check for tree crowns using different radiuses and select the best ones. 
                    Different radiuses allow for integration and merging. Not done yet
    plot: BOOL, whether to plot the output image after each step
    verbose: BOOL to skip comments 
    
    Returns
    -------
    geoPandas DataFrame with detected tree crow position and radius
    
    """
    
    pixel_to_meter = meta_data['transform'][0]
    trees = SAT_map * tree_mask
    
    if plot:
        plt.figure(); plt.imshow(trees)
        plt.axis('off')
    
    if scale_analysis:
        # Scale analysis: Just to design the Gaussian filter, to determine the dominant sizes of target tree crowns
        
        # Different families of tree crowns: e.g.,  radius: 4 -> 4 meters crown diameter
        radius_list = list(range(1,10)) 
        
        means = []
        image_gray = rgb2gray(trees)
        for i in range(0, len(radius_list)-1):
            print(str(radius_list[i+1]) + " - " + str(radius_list[i]))
            means.append( np.mean(morphology.opening(image_gray, selem = morphology.disk(radius = radius_list[i+1])) - 
            morphology.opening(image_gray, selem = morphology.disk(radius = radius_list[i])) ) )
        plt.plot(radius_list[:-1], means)
        plt.xlabel("Diameter of the SE")
        
    if verbose:   
        print("--Extracting tree crowns-- Radius: (px) " + str(crown_element))
    
    # 1- Filtering
    if verbose:
        print("--filtering")
    # Blur the image with gaussian noise, depending on the crown radius level
    image_gray = rgb2gray(trees)
    sigma = 0.3 * crown_element
    image_filtered = gaussian(image_gray, sigma)
    if plot:
        plt.figure(); plt.imshow(image_filtered, cmap = 'gray')
        # plt.title("Filtered image")
        plt.axis('off')
        
        
    # 2- Watershed segmentation    
    if verbose:
        print("--watershed segmentation")
    # Watershed segmentation, the local minima of the image are used as markers.   
    markers_bool  = morphology.local_minima(invert(image_filtered), connectivity = 1) * tree_mask[:,:,0]
    footprint = ndi.generate_binary_structure(markers_bool.ndim, connectivity = 1)
    markers = ndi.label(markers_bool, structure = footprint)[0]     
    segments_watershed = watershed(invert(image_filtered), markers = markers, mask = tree_mask[:,:,0])
    if plot:   
        plt.figure(); plt.imshow(mark_boundaries(trees, segments_watershed), cmap = 'gray')
        # plt.title("segments_watershed")
        plt.axis('off')
     
    # 3- Export boundaries as shapely.Polygons
    if verbose:
        print("--Generate geometries")
    # boundaries = (find_boundaries(segments_watershed)*255).astype('uint8')
    # peaks = np.clip(markers, 0, 1) *255
    # geo_utils.export_GEOtiff("boundaries.tif", boundaries, meta_data)
    # ges.geo_utils.export_GEOtiff("peaks.tif", peaks, meta_data)
    
    centroids = []    
    estimated_crown_radius = []
    label_image = label(segments_watershed)
    for region in regionprops(label_image):
        if region.eccentricity < 0.90:
            centroids.append(region.centroid)
            estimated_crown_radius.append(region.equivalent_diameter/2 * pixel_to_meter)  # radius in meters
            
    rows = []
    cols = []
    
    # Collect centroids of all regions
    for x, y in centroids:
        rows.append(x)
        cols.append(y)
    rows = np.round(np.asarray(rows))
    cols = np.round(np.asarray(cols))
    
    if plot: 
        plt.figure();
        plt.imshow(invert(image_filtered), cmap = 'gray')
        plt.axis('off')
        plt.scatter(cols, rows, s = 20, c = 'red')
        
        
        plt.figure(); plt.imshow(mark_boundaries(trees, segments_watershed), cmap = 'gray')
        plt.axis('off')
        plt.scatter(cols, rows, s = 20, c = 'red')
    
    # if there are no crowns (empty dataframe) there are problems. Handle this case
    if len(rows) != 0:
        assert len(rows) == len(cols)
        # Transform from pixel to geo-referenced coordinates
        pointX = cols * meta_data['transform'][0] + meta_data['transform'][2]
        pointY = rows * meta_data['transform'][4] + meta_data['transform'][5]
        pointXY = (pointX, pointY)                         
        
        # Create a DataFrame to store the information                                            
        df1 = pandas.DataFrame({'pointX': pointXY[0], 'pointY':pointXY[1],
                            'crown_radius' : estimated_crown_radius,                       
                           })
        
        # Convert DataFrame to geoDataFrame
        df1['geometry'] = list(zip(df1['pointX'], df1['pointY']))
        df1['geometry'] = df1['geometry'].apply(Point)
        
        gdf1 = gpd.GeoDataFrame(df1, geometry='geometry', crs = meta_data['crs']) # 32632;  25832, NTM:27700
        
        # Create circles with radius from Point    
        gdf1['geometry'] = gdf1.apply(lambda row: Point(row['geometry']).buffer(row['crown_radius']), axis=1)
        
        if save_output:
            # Export the geoDataFrame as geopackage
            gdf1.to_file(output_filename, driver="GPKG")
    else:
        if verbose:
            print("no crowns detected")
        gdf1 = None
    return gdf1        
        
        
        
def integrate_crowns_2scale(df_large, df_small, overlapping_threshold = 0.8, concatenate = True, save_output = False):
    """
    Integrate two crowns generated at different scales using the <extract_tree_crown> function.
    The integration is done such that the small crowns inside or sufficiently overlapping to large crowns are removed
    
    Parameters
    ----------
    df_large: geopandas DataFrame for the extracted crowns at larger scale
    df_small: geopandas DataFrame for the extracted crowns at smaller scale
    
    Returns
    -------
    geoPandas DataFrame containing the merged cronws
    """
    
    def area_overlapping(distance, r1, r2):
        
        area_small = np.pi * r2**2
        
        if r2 >= r1:
            # it may happen that the circle at lower scale is larger than the one at larger scale.
            # then we keep it
            area = 0        
        else:
            # Here r2 are smaller than r1
            assert r2 < r1
            if distance <= abs(r2 - r1):
                # One circle is inside the area
                area = ( np.pi * min( r1**2, r2**2) )        
            else:
                if distance - r1 > 0:
                    # The center of the small circle is outside the large circle. 
                    # The area of intersection can be computed using circular sectors
                    # Area calculation: https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/
                    x = (r1**2 - r2**2 + distance**2) / (2 * distance)
                    y = (r1**2 - x**2)**0.5
                    area = (r1**2 * np.arcsin(y / r1) + r2**2 * np.arcsin(y / r2) - (x*y + (distance - x)*y)) 
                else:
                    # The center of the small circle is inside the large circle.
                    # We need to use another formula
                    theta_A = 2*np.arccos( (r1**2 + distance**2 - r2**2)/(2*r1*distance) )
                    theta_B = np.arccos(1- (r1/r2)**2 *(1-np.cos(theta_A)) )
                    area = area_small - 0.5*r2**2*(theta_B - np.sin(theta_B))
                
        return (area/area_small)
    
    
    if (df_large is not None) and (df_small is not None): 
        centers_large = df_large[['pointX', 'pointY']].to_numpy()
        centers_small = df_small[['pointX', 'pointY']].to_numpy()
    
        r_large = df_large[['crown_radius']].to_numpy()
        r_small = df_small[['crown_radius']].to_numpy()
        
        
        lines_to_drop = []
        # Take the first large circle, and compute the distance to the small circles. 
        D = distance_matrix(centers_large, centers_small)
       
        
        for index in range(0, len(df_large)):
            # Take only the small circles inside the large circle
            elements_to_consider = np.where(D[index,:] < r_large[index])
            elems_small = centers_small[elements_to_consider]
            
            
            # For each small circle, compute overlapping area between large and small circle
            for index_circle_small in range(0, len(elems_small)):
                    
                r1 = r_large[index] # radius of large circle
                r2 = r_small[elements_to_consider[0][index_circle_small]]
                distance = D[index, elements_to_consider[0][index_circle_small]]
                
                # Overlapping area
                area = area_overlapping(distance, r1, r2)
                if area > overlapping_threshold:
                    # drop the small circle: 
                    # Store the line to remove from the database
                    lines_to_drop.append(elements_to_consider[0][index_circle_small])
                else:
                    # Keep it
                    pass   
        
        # Remove rows from df_small
        df_small = df_small.drop(lines_to_drop)
        
        if concatenate:
            # Merge DataFrames
            df_small = pandas.concat( [df_large, df_small], ignore_index = True)
        if save_output:
            df_small.to_file("_data/NTM dataset/crown_shapes/tmp.gpkg", driver="GPKG") 
    else:
        print("Crowns no detected. Integration not possible")
    return df_small          
        
        

def refine_crowns(df_crowns, meta_data, CROWN_MIN_RADIUS = 1, MIN_DISTANCE = 2, verbose = False):
    """
    Refine the DataFrame
    """  
    if verbose:
        print("-remove crowns smaller than a certain radius ({} meter)".format(CROWN_MIN_RADIUS))
    df_crowns.drop(df_crowns[df_crowns['crown_radius'] < CROWN_MIN_RADIUS].index, inplace = True)
    df_crowns.reset_index(inplace = True, drop=True)
    
    if verbose:
        print("-remove crowns where the centers are very close")
    C1 = df_crowns[['pointX', 'pointY']].to_numpy()
    D = distance_matrix(C1, C1)
    
    # Since D is symmetric, we set the upper-triangular matrix to NaN, so we don't consider that
    D = D * np.tri(*D.shape).T
    D[np.arange(D.shape[1])[:,None] > np.arange(D.shape[0])] = np.nan
    # If two crowns have the same exact center, the distance is zero, we need to remove one:
    np.fill_diagonal(D, np.nan)   
    index_to_drop = list(np.where( (D < MIN_DISTANCE) )[1]) 
    df_crowns = df_crowns.drop(index_to_drop)
    return df_crowns 

    # Since D is symmetric, we set the upper-triangular matrix to NaN, so we don't consider that
    # D = D * np.tri(*D.shape)
    # index_to_drop = list(np.where( (D < MIN_DISTANCE) & (D > 0) )[0]) 
    # df_crowns = df_crowns.drop(index_to_drop)
    # # df_crowns = integrate_crowns_2scale(df_crowns, df_crowns, overlapping_threshold = 0.80, concatenate = False, save_output = False)
    # return df_crowns      
        
 
            
def extract_tree_crown_multi_scale(SAT_map, tree_mask, meta_data = None, 
                                   plot = False, verbose = True, refinement = True,
                                   save_output = True, output_filename = "tree_crowns.gpkg"):

    """ 
    We assume the radius of tree crowns elements are generally:
        1 m (small crowns), 
        2 m (medium size crowns) and 
        4 m (large size crowns)
    Therefore, we set the size of the structuring element (SE) of the morphology operation equal to these values,
    converted in pixels given the resolution of the image (through the meta_data Affine matrix)
    """
    pixel_to_meter = meta_data['transform'][0]
    crown_radius_list = [1,2] # List of crown template radiuses to look for (in meters)  [1,2,3]
    
    # Create a list of radiuses (in pixel) as structuring element in the filtering process
    crown_elements = list(reversed([int(crown_radius / pixel_to_meter) for crown_radius in crown_radius_list]) ) #1,3,5 // 1,2,4
    
    # Compute the tree crowns at the larger radius
    df_large = extract_tree_crown(SAT_map, tree_mask, crown_element = crown_elements[0], 
                                     meta_data = meta_data, verbose = verbose, save_output = False)
    
    # For each crown structuring element (=scale)
    for radius_index in range(1, len(crown_elements)):       
        df_small = extract_tree_crown(SAT_map, tree_mask, crown_element = crown_elements[radius_index], 
                                         meta_data = meta_data, verbose = verbose, save_output = False)

        # Merge them two by two
        if verbose:
            print("Merging crown scales {} and {}".format(crown_elements[radius_index-1], crown_elements[radius_index]))
        df_large = integrate_crowns_2scale(df_large, df_small)
    
        
    # TODO: Set a maximum number for the crown radius.
    if refinement:
        if df_large is not None:
            if verbose:
                print("Refinement")
            df_large = refine_crowns(df_large, meta_data, CROWN_MIN_RADIUS = 1, MIN_DISTANCE = 2, verbose = verbose)
    if save_output:
        if df_large is not None:
            if verbose:
                print("Exporting crowns delineation to: " + output_filename)
            df_large.to_file(output_filename, driver="GPKG")
        else: 
            print("crown = None. Saving not possible") 
    return df_large
        
    
        
def add_tree_species(df_crowns, tree_species_map_file):
    """Add tree species as an additional field in the df_crowns (pandas DataFrame)"""
    
    src = rasterio.open(tree_species_map_file)
    pts = df_crowns[['pointX', 'pointY']] 
    pts.index = range(len(pts))  
    coords = [(x,y) for x, y in zip(pts.pointX, pts.pointY)]
    # Sample the raster at every point location and store values in DataFrame
    df_crowns['tree_species'] = [x[0] for x in src.sample(coords)]
    src.close()
    
    # Sometimes the tree_species_map don't have a specie label when a tree crown is detected.
    # This is because the <TreeSpecieClassifier> is trained separately from the <TreeSegmenter>, and sometimes
    # it misses some trees. 
    # Therefore, tree crowns without a specie label are assigned a species based on the surrouding species.
    
    valid_points = df_crowns[df_crowns['tree_species'].isin([1, 2, 3])]    
    
    if not len(valid_points.index) == 0:
        invalid_points = df_crowns[df_crowns['tree_species'] == 0]
        # Build a KD-tree using the coordinates of the valid points
        from scipy.spatial import cKDTree
        valid_tree = cKDTree(valid_points.geometry.centroid.apply(lambda geom: (geom.centroid.x, geom.centroid.y)).tolist())
    
        # For each invalid point, find the closest valid point using the KD-tree and assign its 'tree_species' value:
        for idx, invalid_point in invalid_points.iterrows():
            nearest_idx = valid_tree.query([invalid_point.geometry.centroid.x, invalid_point.geometry.centroid.y], k=1)[1]
            nearest_valid_point = valid_points.iloc[nearest_idx]
            df_crowns.at[idx, 'tree_species'] = nearest_valid_point['tree_species']
            
            
        replacement_mapping = {1: 'spruce', 2: 'pine', 3: 'deciduous'}
    
        # Use the replace method to rename values
        df_crowns['tree_species'] = df_crowns['tree_species'].replace(replacement_mapping)
        
    else:
        # if there are no trees with labels there
        # Assign tree_specie as "deciduous"
        # TODO: In future, we can think to just delete those trees
        # Also, having the model <TreeSpecieClassifier> to detect trees
        # withjout the external <TreeSegmenter> would solve the problem
        df_crowns['tree_species'] = "deciduous"
    
    return df_crowns


def add_tree_height(df_crowns, nDSM):
    """Add tree height as an additional field in the df_crowns (pandas DataFrame)"""
    
    src = rasterio.open(nDSM)
    pts = df_crowns[['pointX', 'pointY']] 
    pts.index = range(len(pts))  
    coords = [(x,y) for x, y in zip(pts.pointX, pts.pointY)]
    # Sample the raster at every point location and store values in DataFrame
    df_crowns['tree_height'] = [x[0] for x in src.sample(coords)]
    src.close()
    
    # Sanity check: clip the height between 1.4 and 40
    df_crowns['tree_height'] = df_crowns['tree_height'].clip(lower = 1.4, upper = 40) 
    return df_crowns


    
  
    
  
def estimate_DBH(df_crowns):
    """
    Estimate the tree's diameter at breast height (DBH) using allometric equations for each species.
    Height should be expressed in meters [m] and the estimated DBH is in meters [cm]. 
    
    -------
    
    So far, I found a model for spruces developed in Sweden:
    https://academic.oup.com/forestry/article/95/5/634/6580516#375802885
    https://www.diva-portal.org/smash/get/diva2:1605781/FULLTEXT01.pdf
    
    and another model developed for Norway using the Norwegian national forest inventory data from NIBIO
    https://www.tandfonline.com/doi/full/10.1080/21580103.2014.957354
    
    The two models differ a bit, and it is plausible: allometric equations are not general relations but depends
    on the geographical area.  Here we use the Norwegian model.
    """
    
    if not 'tree_species' in df_crowns:
        raise AttributeError("tree species does not exist")
        
    def _DBH_to_height(DBH, b1, b2, const):
        H = ( DBH / (b1 + b2*DBH) )**3 + const 
        return H
    
    def _height_to_DBH(H, b1, b2, const):            
        # Need to be careful about the asymptote at "(1/b2)**3 + const" in the allometric curve.
        # Then the estimated DBH will go to infinity. To prevent that, we clip the height to 0.5 meter before the asymptote
        if H >= (1/b2)**3 + const:
            H = (1/b2)**3 + const - 0.5
            
        k = np.cbrt(H - const)
        DBH =  (b1 * k) / (1 - b2*k)
        # DBH = np.clip( (b1 * k) / (1 - b2*k), a_min =  0, a_max = None)                
        return DBH
            
    
    def _calculate_DBH(row):
        if row['tree_species'] == "spruce": 
            DBH = _height_to_DBH(row['tree_height'], b1 = 2.2131 , b2 = 0.3046, const = 1.3)
        elif row['tree_species'] == "pine": 
            DBH = _height_to_DBH(row['tree_height'], b1 = 2.2845 , b2 = 0.3318, const = 1.3)
        elif row['tree_species'] == "deciduous": 
            DBH = _height_to_DBH(row['tree_height'], b1 = 1.649 , b2 = 0.373, const = 1.3)
        else:
            DBH = np.nan
        return DBH

    
    df_crowns['DBH'] = df_crowns.apply(_calculate_DBH, axis=1)
    
    return df_crowns




def calculate_Hegyi_index():
    # TODO in future
    pass



def calculate_gap_factor(trees_within_corridor: pandas.DataFrame, crowns: pandas.DataFrame, distance_to_consider = 10):
    """
    Calculate the gap coefficient for every tree inside the corridor (passed as input).
    Because the gap_coeff is calculated in the sourriding area, some trees that may shield the trees inside the corridor 
    can be outside of the corridor. That's why the crowns dataframe is passed as well.
    
    --> There are many ways to implement this gap factor. 
    - Approach 1: We consider the distance to the closest tree 
    whose height is at least the height of the tree for which we want to calcualte the gap factor.
    -Approach 2: We consider just the presence of a tree
    
    Parameters
    ----------
    trees_within_corridor: geopandas DataFrame for trees inside the corridor
    crowns: geopandas DataFrame for trees in the large window extracted around the infrastructure line
    distance_to_consider: deafault = 10. Distance for a tree to be considered neighbor
    
    Returns
    -------
    geoPandas DataFrame with the gap coefficient for each direction (North, North-East, etc...) as additional fields
    """
    
    def gap_coeff_per_direction(tree: pandas.Series, sector_df : pandas.DataFrame, approach = 2):
        """ 
        tree: tree under consideration
        sector_df: tree inside a circular sector around <tree>
        """
        
        D = distance_matrix(
            tree[['pointX', 'pointY']].to_numpy(dtype = 'float32').reshape(1,2), 
            sector_df[['pointX', 'pointY']].to_numpy(dtype = 'float32')).reshape(-1,)
        
        # Calculate the distance to the closest tree with height equal or higher than the considered tree
        large_trees_distance = D[tree['tree_height'] <= sector_df['tree_height']]
        
        if approach == 1:
            #--- APPROACH 1: we consider the distance to the closest tree 
            # whose height is at least the height of the tree for which we want to calcualte the gap factor        
            if len(large_trees_distance) != 0:
                    f_gap = min(large_trees_distance) /2 # divide by 2 to make it in the range (0,5)
                    f_gap = np.clip(f_gap, a_min = 1, a_max = 5) # ensure the range is between 0 and 5
            else:
                f_gap = 5
                
        elif approach == 2:                    
            #--- APPROACH 2: we consider just the presence of a tree in the sector 
            if len(large_trees_distance) != 0:
                f_gap = 1
            else:
                f_gap = 2
            
        return f_gap   
    
    
    def _assign_wind_rose_category(angle):
        if -22.5 <= angle <= 22.5 or angle >= 337.5 or angle <= -337.5:
            return "E"
        elif 22.5 < angle <= 67.5:
            return "NE"
        elif 67.5 < angle <= 112.5:
            return "N"
        elif 112.5 < angle <= 157.5:
            return "NW"
        elif 157.5 < angle <= 202.5:
            return "W"
        elif 202.5 < angle <= 247.5:
            return "SW"
        elif 247.5 < angle <= 292.5:
            return "S"
        elif 292.5 < angle <= 337.5:
            return "SE"
        else:
            return "Undefined"  # Handle angles outside the specified range
    
    sectors_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    # Iterate through each tree in trees_within_corridor
    for tree_index, tree in trees_within_corridor.iterrows():
      
        # compute the distances from all the other trees
        distance_tree_crowns = distance_matrix(
            tree[['pointX', 'pointY']].to_numpy(dtype = 'float32').reshape(1,2), 
            crowns[['pointX', 'pointY']].to_numpy(dtype = 'float32'))
        
        # Select only the trees in the neighborhood within a certain distance from the considered tree
        tree_neighborhood = crowns[distance_tree_crowns.T < distance_to_consider]
        
        # Remove the considered tree from the neighborhood
        tree_neighborhood = tree_neighborhood.drop(index=tree_index, errors='ignore')
   
        # Compute angles
        angles = (np.rad2deg(np.arctan2(tree_neighborhood['pointY'] - tree['pointY'], tree_neighborhood['pointX'] - tree['pointX'])) + 360)%360  
        
        # Cluster angles        
        wind_rose_categories = angles.apply(_assign_wind_rose_category)
        
        # To avoid warning: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        # pandas.options.mode.chained_assignment = None  # default='warn'
        tree_neighborhood['Sector'] = wind_rose_categories
        
        # Initialize the shielding coefficients for each sector
        shielding_coefficients = {}
    
        # Iterate through sectors and filter the DataFrame for each sector
        for sector in sectors_labels:
            sector_df = tree_neighborhood[tree_neighborhood['Sector'] == sector]            
            # Compute gap factor
            f_gap = gap_coeff_per_direction(tree, sector_df)
            shielding_coefficients[sector] = f_gap
            
        # Add the shielding coefficients to the trees_within_corridor DataFrame
        for sector in sectors_labels:
            column_name = "f_gap_" + str(sector)
            trees_within_corridor.at[tree_index, column_name] = shielding_coefficients.get(sector, np.nan)
    return trees_within_corridor
    


def plot_crowns(df_crowns, SAT_image = None, meta_data = None):
    
    # Plot only if the number is not too large
    if len(df_crowns.index) < 3000:
        
        if SAT_image is None:
            # plot just the crowns
            plt.scatter(df_crowns['pointX'], df_crowns['pointY'])
        else:
            # plot crowns on top of the SAT image
            plt.imshow(geo_utils.multiband_to_RGB(SAT_image))
            pointX = ((df_crowns['pointX'].to_numpy() - meta_data['transform'][2])/meta_data['transform'][0] ).astype(int)
            pointY = ((df_crowns['pointY'].to_numpy() - meta_data['transform'][5])/meta_data['transform'][4] ).astype(int)
            plt.scatter(pointX, pointY, color = 'red')
    else:
        print("too many crowns")
        
        
        
def calculate_critical_wind_speed_breakage(trees_within_corridor: pandas.DataFrame, wind_direction = 'East'):
     
    if not 'tree_species' in trees_within_corridor:
        raise AttributeError("tree species does not exist")
    if not 'tree_height' in trees_within_corridor:
        raise AttributeError("tree height does not exist")
        
    # Select the correct f_gap_coefficient based on the actual wind direction
    sector = wind_direction # to be changed accorind to weather data
    
    def _add_MOR(df_crowns):
        """ 
        Assign modulus of rupture (MOR) to different species. MOR are expressed in MPa.
        Values are taken from: https://www.wood-database.com/
        """
        df_crowns['MOR'] = 0.0
        df_crowns.loc[df_crowns['tree_species'] == "spruce", 'MOR'] = 63 * 10**6
        df_crowns.loc[df_crowns['tree_species'] == "pine", 'MOR'] = 83.3 * 10**6
        df_crowns.loc[df_crowns['tree_species'] == "deciduous", 'MOR'] = 114.5 * 10**6
        return df_crowns 
    
    def _add_Tc(df_crowns_row):
        """ Calculate the turning coefficient Tc
        https://www.sciencedirect.com/science/article/pii/S1364815213002090
        Possible future extension is to calculate Tc including the Hegyi competion index 
        """
        cm_to_m = 0.01
        Tc = 117 * (cm_to_m * df_crowns_row['DBH'])**2 * df_crowns_row['tree_height'] 
        return Tc

           
    def _critical_wind_speed_breakage(df_crowns_row):
        cm_to_m = 0.01
        f_CW = 1.25 #additional moment provided by the overhanging displaced mass of the canopy 
        crit_wspeed_breakage = np.sqrt( (df_crowns_row['MOR'] * (cm_to_m * df_crowns_row['DBH'])**3 * np.pi) / (32 * df_crowns_row['Tc'] * df_crowns_row['f_gap_' + sector] * f_CW)  )
        return crit_wspeed_breakage

        
    # Add modulus of rupture
    trees_within_corridor = _add_MOR(trees_within_corridor)
    
    # Calculate turning coefficient
    trees_within_corridor['Tc'] = trees_within_corridor.apply(_add_Tc, axis=1)
    
    # Calculate critical wind speed for breakage
    trees_within_corridor['Crit_wspeed_breakage'] = trees_within_corridor.apply(_critical_wind_speed_breakage, axis=1)
    
        
    return trees_within_corridor
        

       
        
        
        
        
        
        
        
        
