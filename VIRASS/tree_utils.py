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
    sigma = 0.5 * crown_element
    image_filtered = gaussian(image_gray, sigma)
    if plot:
        plt.figure(); plt.imshow(image_filtered, cmap = 'gray')
        plt.title("Filtered image")
        
        
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
        plt.title("segments_watershed")
     
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
    crown_radius_list = [1,2,3] # List of crown template radiuses to look for (in meters) 
    
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
    """Add tree species as an additional field in the df_crowns (pandas DatasFrame)"""
    
    src = rasterio.open(tree_species_map_file)
    pts = df_crowns[['pointX', 'pointY']] 
    pts.index = range(len(pts))  
    coords = [(x,y) for x, y in zip(pts.pointX, pts.pointY)]
    # Sample the raster at every point location and store values in DataFrame
    df_crowns['tree_species'] = [x[0] for x in src.sample(coords)]
    src.close()
    
    # Sometimes the tree_species_map donn't have a specie label if if a tree crown is detected.
    # This is because the <TreeSpecieClassifier> is trained separately from the <TreeSegmenter>, and sometimes
    # it misses some trees. 
    # Therefore, tree crowns without a specie label are assigned a species based on the surrouding species.
    # TODO: Assign a tree specie
    return df_crowns


def add_tree_height(df_crowns, nDSM):
    """Add tree height as an additional field in the df_crowns (pandas DatasFrame)"""
    
    src = rasterio.open(nDSM)
    pts = df_crowns[['pointX', 'pointY']] 
    pts.index = range(len(pts))  
    coords = [(x,y) for x, y in zip(pts.pointX, pts.pointY)]
    # Sample the raster at every point location and store values in DataFrame
    df_crowns['tree_height'] = [x[0] for x in src.sample(coords)]
    src.close()
    return df_crowns
       
        
        
        
        
        
        
        
        
