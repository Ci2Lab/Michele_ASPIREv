import numpy as np
import rasterio

# Optional dependency:
    #laspy is required to work with this module, but in terms of the general usage of 
    # storm-ASPIREv library, it is not required
try:
    import laspy
except ImportError:
    laspy = None

from scipy.interpolate import griddata, Rbf
from scipy.ndimage import median_filter, distance_transform_edt, generic_filter

from . import utils
from . import io


@utils.measure_time
def create_dsm(laz_file : str, grid_size = 1.0, 
               fill_holes = False, filling_method = 'linear',
               remove_spikes = False, kernel_size = 3):
    """
    A DSM represents the earth's surface and includes all objects on it. 
    The process involves projecting the point cloud onto a 2D grid and finding 
    the highest point (in terms of elevation) within each cell of the grid.
    """
    # Load the .laz or .las file
    with laspy.open(laz_file) as file:
        las = file.read()
        points = np.vstack((las.x, las.y, las.z)).transpose()

    # Define the grid
    min_x, min_y, max_x, max_y = np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])
    cols = int((max_x - min_x) / grid_size) + 1
    rows = int((max_y - min_y) / grid_size) + 1

    # Initialize the DSM array with NaNs
    dsm = np.full((rows, cols), np.nan)

    # Assign points to grid cells and find the highest point per cell
    for point in points:
        col = int((point[0] - min_x) / grid_size)
        row = int((point[1] - min_y) / grid_size)
        z = point[2]
        if np.isnan(dsm[row, col]) or dsm[row, col] < z:
            dsm[row, col] = z
    
    """ 
    The generated dsm appears upside down, this issue often relates to 
    how the y-coordinates are interpreted. This problem arise during the 
    rasterization process, where the convention for image coordinates 
    (origin at the top left) conflicts with the typical GIS coordinate 
    system (where the y-coordinate increases northward).
    --> Here, we flip the y-axis
    """        
    dsm = np.flipud(dsm)
    
    if fill_holes:
        """
        Filling holes in a Digital Surface Model (DSM) is a common task,
        aimed at correcting gaps or voids that can occur due to various reasons 
        such as occlusion, low point density, or data collection errors.
        """
        print(f"-- Filling holes: method: `{filling_method}`")
        if filling_method == 'linear':
            dsm = fill_holes_linear(dsm)
        elif filling_method == 'rbf':
            dsm = fill_holes_rbf(dsm)
        else:
            print(f"Method: `{filling_method}` not implemented. \n--Skipping")
            pass
        
    if remove_spikes:
        print(f"-- Removing spikes: kernel size: {kernel_size}")
        dsm = remove_spikes_median_filter(dsm, kernel_size)
                

    return dsm



@utils.measure_time
def create_dtm(laz_file : str, grid_size = 1.0, 
               fill_holes = False, filling_method = 'linear'):
    """ It assumes that the .laz/.las file contains ground points classified 
    with the class code 2, which is standard for ground points in LAS files.
    Refer to https://www.asprs.org/wp-content/uploads/2010/12/LAS_Specification.pdf
    for las specifications
    """
    with laspy.open(laz_file) as file:
        las = file.read()
        
        # Filter for ground points
        ground_points = las.points[las.classification == 2]
        coordinates = np.vstack((ground_points.x, ground_points.y, ground_points.z)).transpose()
    
    # After extracting the ground points, we can interpolate these points 
    # to create a grid that represents the DTM.
    min_x, min_y, max_x, max_y = np.min(coordinates[:,0]), np.min(coordinates[:,1]), np.max(coordinates[:,0]), np.max(coordinates[:,1])
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_size, min_y:max_y:grid_size]

    
    # Interpolate z values
    dtm = griddata(coordinates[:, :2], coordinates[:, 2], (grid_x, grid_y), method = 'linear')

    
    """
    The dtm here appears rotated when compared to its representation in QGIS. 
    This could due to how the coordinate system or the array indices are 
    interpreted between the DTM creation process and QGIS.
    Similar issue as to the `create_dsm`, when np.flipud did the job
    """
    dtm = np.rot90(dtm)
    
    if fill_holes:
        """
        Filling holes in a Digital Surface Model (DSM) is a common task,
        aimed at correcting gaps or voids that can occur due to various reasons 
        such as occlusion, low point density, or data collection errors.
        """
        print(f"-- Filling holes: method: `{filling_method}`")
        if filling_method == 'linear':
            dtm = fill_holes_linear(dtm)
        elif filling_method == 'rbf':
            dtm = fill_holes_rbf(dtm)
        else:
            print(f"Method: `{filling_method}` not implemented. \n--Skipping")
            pass
        
    return dtm



def build_meta_data(laz_file_path : str, dsm : np.array, grid_size) -> dict:
    """
    Create meta_data from the a .laz/.las file (to get its header), the generated dsm 
    and grid size used to generate the dsm through the `create_dsm` function.
    """
    # get crs
    with laspy.open(laz_file_path) as file:
        crs = file.header.parse_crs().to_epsg()        
        header = file.header        
        
        """
        To extract the top_left_x and top_left_y coordinates, we use the minimum 
        X and maximum Y coordinates from the file's header. 
        These minimum coordinates essentially represent the bottom-left corner of 
        the bounding box of your point cloud data.
        """
        # Extracting the minimum X and the maximum Y for the top-left corner
        top_left_x = header.min[0]  # min X
        top_left_y = header.max[1]  # max Y    
        
        
    # Define the transform
    transform = rasterio.transform.from_origin(west = top_left_x,  
                                               north = top_left_y, # + dsm.shape[0] * grid_size, 
                                               xsize = grid_size, 
                                               ysize = grid_size)
    meta_data = {'driver' : 'GTiff',
            'height' : dsm.shape[0],
            'width' : dsm.shape[1],
            'count' : 1,
            'dtype' : str(dsm.dtype),
            'crs' : rasterio.crs.CRS.from_epsg(str(crs)),
            'transform' : transform}    
    return meta_data
    


# --- NOT TESTED ---
@utils.measure_time
def merge_laz_files(input_files : list, output_file : str):
    # List to hold all points from all files
    all_points = []
    
    # Attributes to be merged; extend as needed
    attributes = ['points', 'X', 'Y', 'Z']
    
    # Load each file and append its points to the list
    for file_name in input_files:
        with laspy.open(file_name) as f:
            las = f.read()
            all_points.append({attr: getattr(las, attr) for attr in attributes})
    
    # Create a new LAS object for the merged point cloud
    header = laspy.LasHeader(version="1.4", point_format=las.header.point_format)
    merged_las = laspy.LasData(header)
    
    # Concatenate attributes from all point clouds
    for attr in attributes:
        setattr(merged_las, attr, np.concatenate([p[attr] for p in all_points]))
    
    # Save the merged LAS file
    with laspy.open(output_file, mode="w") as f:
        f.write(merged_las)    
  
        
     
def fill_holes_linear(dsm : np.array):
    assert len(dsm.shape) == 2
    # Create a meshgrid of x, y coordinates
    x = np.arange(0, dsm.shape[1])
    y = np.arange(0, dsm.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    # Mask for valid (non-NaN) and invalid (NaN) points
    valid_mask = ~np.isnan(dsm)
    invalid_mask = np.isnan(dsm)
    
    # Coordinates of valid and invalid points
    valid_coords = np.array((xx[valid_mask], yy[valid_mask])).T
    invalid_coords = np.array((xx[invalid_mask], yy[invalid_mask])).T
    valid_values = dsm[valid_mask]
    
    # Interpolate using linear method
    dsm_filled = dsm.copy()
    dsm_filled[invalid_mask] = griddata(valid_coords, valid_values, invalid_coords, method='linear')
    
    # Find remaining NaNs after interpolation
    remaining_nan_mask = np.isnan(dsm_filled)
    
    if np.any(remaining_nan_mask):
        # Use distance transform to find the nearest non-NaN values
        distances, indices = distance_transform_edt(remaining_nan_mask, return_distances=True, return_indices=True)
        dsm_filled[remaining_nan_mask] = dsm_filled[tuple(indices[i][remaining_nan_mask] for i in range(dsm.ndim))]
    
    return dsm_filled 

   
def fill_holes_rbf(dsm):
    x = np.arange(0, dsm.shape[1])
    y = np.arange(0, dsm.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    valid_mask = ~np.isnan(dsm)
    invalid_mask = np.isnan(dsm)
    
    valid_coords = np.array((xx[valid_mask], yy[valid_mask])).T
    invalid_coords = np.array((xx[invalid_mask], yy[invalid_mask])).T
    valid_values = dsm[valid_mask]
    
    # Use RBF interpolation
    rbf_interpolator = Rbf(valid_coords[:, 0], valid_coords[:, 1], valid_values, function='linear')
    dsm_filled = dsm.copy()
    dsm_filled[invalid_mask] = rbf_interpolator(invalid_coords[:, 0], invalid_coords[:, 1])
    
    return dsm_filled


def remove_spikes_median_filter(dsm, kernel_size=3):
    """
    Removes spikes from a DSM using median filtering.
    
    Parameters:
        dsm (numpy.ndarray): The DSM array with potential spikes.
        kernel_size (int): The size of the moving window to apply the median filter. 
                            This value determines the area around each pixel to 
                            consider for filtering. A larger value will
                           be more effective at removing larger spikes 
                           but may also blur the data more.
    
    Returns:
        numpy.ndarray: The DSM array with spikes removed.
    """
    # Apply median filter
    dsm_filtered = median_filter(dsm, size=kernel_size)
    
    return dsm_filtered