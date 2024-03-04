"""
@author: Michele Gazzea
"""
import matplotlib.pyplot as plt
from osgeo import ogr, osr, gdal
import numpy as np
from scipy import ndimage
from skimage import morphology 
import os
import rasterio
from affine import Affine
from shapely.geometry import shape, box
import geopandas
import gc
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.windows import Window

# Optional dependency:
    #cv2 is required to work with this the pansharpen module, but in terms of the general usage of 
    # storm-ASPIREv library, it is not required.
    # TODO: In future, it could be replace by scipy and remove completely the dependency.
try:
    import cv2
except ImportError:
    cv2 = None
    
from . import utils
from . import io
  

def imshow(SAT_image):
    SAT_image = multiband_to_RGB(SAT_image)
    plt.figure()
    plt.imshow(SAT_image)
    
    
def multiband_to_RGB(SAT_image, verbose = True):
    """
    Take the RGB components of a satellite multi-channel image
    """
    
    # Ensure the bands are in the last axis
    SAT_image = utils.convert_to_channel_last(SAT_image, verbose = verbose)
    number_of_channels = SAT_image.shape[-1] 
    if verbose:
        print("Number of bands: {}".format(number_of_channels))

    if number_of_channels == 8: 
        SAT_image = np.take(SAT_image, indices = [4,2,1], axis = -1)
    elif number_of_channels == 4:
        SAT_image = np.take(SAT_image, indices = [0,1,2], axis = -1)
    else:
        print("Provide indices for Red:")
        RED_index = input()
        print("Provide indices for Green:")
        GREEN_index = input()
        print("Provide indices for Blue:")
        BLUE_index = input()
        SAT_image = np.take(SAT_image, indices = [RED_index,GREEN_index,BLUE_index], axis = -1) 
    return SAT_image 
    

def compute_ndvi(satellite_image: np.array):
    assert utils.is_channel_last(satellite_image)
    
    satellite_image = satellite_image.astype('float32')
    
    if satellite_image.shape[-1] == 8:
        nir_index = 6
        red_index = 4
    elif satellite_image.shape[-1] == 4:
        nir_index = 4
        red_index = 1
    else:
        print("Provide indices for <infrared> band:")
        nir_index = input()
        print("Provide indices for <red> band:")
        red_index = input()
        
    nir = np.take(satellite_image, indices = nir_index, axis = -1)
    red = np.take(satellite_image, indices = red_index, axis = -1)       
    ndvi = (nir - red) / (nir + red + 1e-10)  # Add a small number to avoid division by zero
    ndvi = np.expand_dims(ndvi, axis=-1) # Add a dimension in the last axis
    return ndvi


def image_enchancement(image, verbose = True):
    """
    Process an image to enhance contrast based on 2-98% percentile (=color balance version) 
    """
    image = utils.convert_to_channel_last(image, verbose = verbose)
    print(image.shape)
    N_bands = image.shape[-1]
    
    # Create an alpha channel based on the sum of pixel values across all bands
    # Set to 1 where sum > 0 (to keep), and 0 where sum = 0 (to discard)
    _sum = np.sum(image, axis=-1, keepdims=True)
    alpha = (abs(_sum) > 0).astype(int)   
    # Repeat the alpha channel for each band to create a mask
    mask = np.repeat(alpha == 0, repeats = N_bands, axis=-1)
    image[mask] = np.nan
       
    for i in range(0, N_bands):
        if verbose:
            print("--Processing band: {} / {}".format(str(i+1), N_bands) )       
        a = np.nanpercentile(image[:,:,i].flatten(), 2)
        b = np.nanpercentile(image[:,:,i].flatten(), 98)
        image[:,:, i] = np.clip( ( ( (image[:,:, i] - a) / (b-a) ) * 255) , 0,255)        
        # image[np.where(image[:,:,i] == np.nan)] = 0
    
    image[np.isnan(image)] = 0
    return image.astype('uint8')
    
 
     
def pre_process(image_src, meta_data = None, save = False, verbose = True):
    """
    Take the 4Bands uint16 raw image and process it to convert it into uint8 image [0-255].
    Cumulative count cut to 2-98% percentile and MinMax stretch is applied
    ---
    Input: path to the SAT image
    Output: preprocessed image  [uint8]
    """

    # if image_src is a path, open the image
    if isinstance(image_src, str):
        print("--Preprocessing image: {}".format(image_src))
        image, meta_data = io.open_geoTIFF(image_src, verbose = verbose)
    else:
        # check if image_src is already an array
        assert isinstance(image_src, np.ndarray)
        image = image_src.copy()
        if meta_data is None:
            raise ValueError("meta_data needs to be provided --is None now")
    
    image = image.astype('float32')
    #--Process the satellite bands to enhance contrast (=color balance version) 
    output = image_enchancement(image, verbose = verbose)
        
    return output, meta_data



def transformGPS_coord(pointXY, outputEPSG = 32632, inputEPSG = 4326 ):
    """ 
    Convert a pointXY = (longitude, latitude = 5.16..., 61.3....) format into another CRS, as default 32632 (EPSG:32632 WGS 84 / UTM zone 32N)
    """
    pointX = pointXY[1] #Latitude
    pointY = pointXY[0] #Longitude
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pointX, pointY)    
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)    
    # transform point
    point.Transform(coordTransform)    
    # print point in EPSG 4326
    return (point.GetX(), point.GetY())



def is_point_valid(point, meta_data, radius = 0):
        """
        #Test if a point coordinate (and the possible buffer within radius) is in the map. 
        meta_data is the geoTiff header info
        """
        u = int((point[0] - meta_data["transform"][2]) / meta_data["transform"][0])
        v = int((point[1] - meta_data["transform"][5]) / meta_data["transform"][4])
        if u >= radius and u <= meta_data["width"]-radius and v >= radius and v <= meta_data["height"]-radius:
           return True
        else:
            return False
        

def extract_window(point, SAT_image, radius, meta_data):
    """

    Parameters
    ----------
    point : TYPE
        DESCRIPTION.
    SAT_image : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    meta_data : TYPE
        DESCRIPTION.

    Returns
    -------
    window : TYPE
        DESCRIPTION.
    meta_data_W : TYPE
        DESCRIPTION.

    """
    assert utils.is_channel_last(SAT_image)
    # Make sure that the point coordinate system is the same as the Map coordinate system
    # Map_CRS = meta_data['crs'].data['init'].split(":")[1]
    # assert Map_CRS == point_CRS
    # TODO: Dealt with different CRS automatically using the 'transformGPS_coord' function 
        
    if is_point_valid(point, meta_data, radius):
        col = int( (point[0] - meta_data['transform'][2]) / meta_data['transform'][0])
        row = int( (point[1] - meta_data['transform'][5]) / meta_data['transform'][4])
        window = SAT_image[row-radius:row+radius, col-radius:col+radius,:]
        
        # Compute the new meta_data associated with the window
        meta_data_W = meta_data.copy()
        meta_data_W['height'] = window.shape[0]
        meta_data_W['width'] = window.shape[1]
        # the offsets of the affine matrix are different
        meta_data_W['transform'] = Affine(meta_data['transform'][0], meta_data['transform'][1], point[0] - radius/2 , \
                               meta_data['transform'][3], meta_data['transform'][4], point[1] + radius/2)
    else:
        print("WARNING: point not valid")
        window = None
        meta_data_W = None
    
    return window, meta_data_W
           

    
def generate_tree_points(nDSM, NDVI, Th1 = 2, Th2 = 0.1):
    """
    Detect pixels belonging to trees from a nDSM and NDVI.
    Trees are calculated setting two different thresholds Th1, Th2 to nDSM and NDVI, respectively.
    """
    
    assert utils.is_channel_last(nDSM) and utils.is_channel_last(NDVI)
    def fix_dimensions(nDSM, NDVI):
        # during preprocessing height/width might be off by one pixel
        if nDSM.shape != NDVI.shape:
            print('Resizing to same shape')
            h, w, _ = [min(s) for s in zip(nDSM.shape, NDVI.shape)]
            nDSM = nDSM[:h,:w, :]
            NDVI = NDVI[:h,:w, :]
        else:
            print("Dimensions are good")   
        return nDSM, NDVI
    
    # remove noise and isolated pixels (i.e. powerlines, birds, bushes, and more)
    def erode_and_dilate(source, structure=np.ones((3,3)), it=1):
        eroded = ndimage.binary_erosion(source, structure=structure, iterations=it).astype(source.dtype)
        dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=it*2).astype(eroded.dtype)
        result = np.logical_and(source,dilated).astype(dilated.dtype)
        return result
    
    def dilate_and_erode(source, structure=np.ones((2,2)), it=1):
        dilated = ndimage.binary_dilation(source, structure=structure, iterations=it*2).astype(source.dtype)
        eroded = ndimage.binary_erosion(dilated, structure=structure, iterations=it).astype(dilated.dtype)       
        result = np.logical_and(source,eroded).astype(eroded.dtype)
        return result
    
    nDSM, NDVI = fix_dimensions(nDSM, np.clip(NDVI, -1,1)) 

    nDSM = np.clip(nDSM, a_min = 0 , a_max = 30)      
    tree_height = nDSM > Th1
    green = NDVI > Th2
    tree_points = np.squeeze(np.logical_and(tree_height, green).astype('uint8'), axis = -1) 
        
    return erode_and_dilate(tree_points, structure=np.ones((2,2)), it=2) *255



def create_alpha_channel(SAT_image, refinement = "closing"):
    assert utils.is_channel_last(SAT_image)
    """ 
    no data have 0 in all channels. 
    This works most of the time but some pixels have 0 values in all channels
    even if they are valid pixels, need to distiguish the outer layer from the region 
    within the satellite. Therefore, we perform a morphological closing on the 
    pre-computed alpha channel 
    
    refinemnt: 
        - `Opening` removes small objects from the foreground of an image, 
    placing them in the background
        - `Closing` removes small holes in the foreground, 
        changing small islands of background into foreground. 
    """    
            
    #alpha_layer = np.clip(np.sum(SAT_image, axis = -1), 0, 1)
    # _sum = np.sum(SAT_image, axis=-1, keepdims=True)
    # alpha_layer = (abs(_sum) > 0).astype(int) 
    alpha_layer = (SAT_image.sum(axis = -1) > 0)

    
    # Perform morphological closing on the pre-computed alpha channel
    
    # WARNING: 'selem' argument is deprecated in favor of 'footprint' in newer versions of skimage!
    # alpha_layer = (np.expand_dims( morphology.binary_closing(alpha_layer, 
    #                               selem = morphology.disk(radius = 3)), 
    #     axis=-1)*255).astype('uint8')
    
    if refinement == "opening":
        alpha_layer = (morphology.binary_opening(alpha_layer, footprint = morphology.disk(radius = 10))*255).astype('uint8')
    
    elif refinement == "closing":
        alpha_layer = (morphology.binary_closing(alpha_layer, footprint = morphology.disk(radius = 10))*255).astype('uint8')

    return alpha_layer



def raster_to_vector(input_raster, meta_data = None, output_vector = "output_vector.gpkg", save = False):
    from rasterio import features
        
    # if input_raster is a path, open the image
    if isinstance(input_raster, str):
        with rasterio.open(input_raster) as src:
            crs = src.crs
            mask = src.read(1)
            shapes = list(features.shapes(mask, transform = src.transform))
    else:
        # check if input_raster is already an array
        assert isinstance(input_raster, np.ndarray)
        if meta_data is None:
            raise ValueError("meta_data needs to be provided --is None now")
        if len(input_raster.shape) == 3:
            mask = input_raster[:,:,0]
        else:
            mask = input_raster
        shapes = list(features.shapes(mask, transform = meta_data['transform']))
        crs = meta_data['crs']
    
    white_shapes = [shape(s[0]) for s in shapes if s[1] == 255]
    gdf = geopandas.GeoDataFrame(geometry=white_shapes)
    gdf.crs = crs  # Set the coordinate reference system
    
    if save:
        gdf.to_file(output_vector, driver="GPKG")
    return gdf


def generate_fake_meta_data(image: np.array):
    from affine import Affine
    meta_data = {'driver': 'GTiff', 
                 'dtype': 'uint8', 
                 'nodata': None, 
                 'width': image.shape[0], 
                 'height': image.shape[1], 
                 'count': image.shape[2], 
                 'crs': None, 
                 'transform': Affine(1, 0, 0, 0, -1, 0), 
                 'tiled': False, 
                 'interleave': 'pixel'}
    return meta_data
    

@utils.deprecated   
def split_raster_to_parts(SAT_image, N, tmp_folder = "tmp", name = "part"):    
    # Create a temporary folder to store the parts of the image    
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    assert utils.is_channel_last(SAT_image)
    
    # Check if the directory is empty or if overwrite is implied by checking 
    # the existence of specific parts
    existing_files = os.listdir(tmp_folder)
    parts_exist = any(name in file for file in existing_files)
    
    # Proceed to split if no parts exist
    if not parts_exist:      
        # Here we split the image by rows, doing it by columns is the same
        rows_per_part = np.ceil(SAT_image.shape[0] / N)
        for i in range(0, N):
            a = int(i*rows_per_part)
            b = int((i+1)*rows_per_part)
            part = SAT_image[a:b,:,:] 
            # Save each part as a numpy array
            part_filename = os.path.join(tmp_folder, f"{name}{i}.npy")
            np.save(part_filename, part)
    else:
        print(f"The image seems to be already divided into chunks. name: {name}")


def split_geoTIFF_to_parts(image_src, meta_data = None, n_splits = 10, tmp_folder = "tmp", name = "part"):
    # Create a temporary folder to store the parts of the image    
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        
    with rasterio.open(image_src) as src:
        width, height = src.width, src.height
                    
        # Calculate width of each split
        split_height  = height // n_splits
        
        for i in range(n_splits):
            # Calculate window. The last split may need to include extra pixels to account for rounding
            if i == n_splits - 1:  # Last split
                window = Window(0, i * split_height, width, height - i * split_height)
            else:
                window = Window(0, i * split_height, width, split_height)
            
            # Read the data for the current window/split
            window_data = src.read(window=window)
            
            # Define the output path for the current split
            output_path = os.path.join(tmp_folder, f"{name}_{i+1}.tif")
            
            # Update the metadata for the split
            out_meta = src.meta.copy()
            out_meta.update({
                'width': window.width,
                'height': window.height,
                'transform': rasterio.windows.transform(window, src.transform)
            })
            
            # Save the split as a new GeoTIFF file
            io.export_geoTIFF(output_path, window_data, out_meta, verbose = False)
                
                
def align_maps(A, B):
    """
    A and B are two images that are supposed to have the same dimensions. 
    Sometimes, the height/width between A and B might be off by one pixel due to some clipping.
    This will perform a small alignment"""
    if A.shape == B.shape:
        print("X_map and y_map have the same dimensions")
    else:
        print("X_map shape {} \ny_map shape {}".format(A.shape, B.shape))
        print('--Resizing to same shape')
        h, w, _ = [min(s) for s in zip(A.shape, B.shape)]
        A = A[:h,:w,:]
        B = B[:h,:w,:]
        print("SAT shape {} \ny_map shape {}".format(A.shape, B.shape))
    return A, B



def pansharpen(m, pan, psh, R = 1, G = 2, B = 3, NIR = 4, method = 'simple_brovey', W = 0.1):
    """ 
    This function is used to pansharpen a given multispectral image using its corresponding panchromatic image via one of 
    the following algorithms: 'simple_brovey, simple_mean, esri, brovey'.
  
    Inputs:
    - m: File path of multispectral image to undergo pansharpening
    - pan: File path of panchromatic image to be used for pansharpening
    - psh: File path of pansharpened multispectral image to be written to file
    - R: Band number of red band in the multispectral image
    - G: Band number of green band in the multispectral image
    - B: Band number of blue band in the multispectral image
    - NIR: Band number of near - infrared band in the multispectral image
    - method: Method to be used for pansharpening
    - W: Weight value to be used for brovey pansharpening methods
  
    Outputs:
    - img_psh: Pansharpened multispectral image
    https://github.com/ThomasWangWeiHong/Simple-Pansharpening-Algorithms/blob/master/Simple_Pansharpen.py
  
    """    
    with rasterio.open(m) as f:
        metadata_ms = f.profile
        img_ms = np.transpose(f.read(tuple(np.arange(metadata_ms['count']) + 1)), [1, 2, 0])
    
    with rasterio.open(pan) as g:
        metadata_pan = g.profile
        img_pan = g.read(1)
    
    
    
    
    ms_to_pan_ratio = metadata_ms['transform'][0] / metadata_pan['transform'][0]
    rescaled_ms = cv2.resize(img_ms, dsize = None, fx = ms_to_pan_ratio, fy = ms_to_pan_ratio, 
                             interpolation = cv2.INTER_CUBIC).astype(metadata_ms['dtype'])

  
    if img_pan.shape[0] < rescaled_ms.shape[0]:
        ms_row_bigger = True
        rescaled_ms = rescaled_ms[: img_pan.shape[0], :, :]
    else:
        ms_row_bigger = False
        img_pan = img_pan[: rescaled_ms.shape[0], :]
        
    if img_pan.shape[1] < rescaled_ms.shape[1]:
        ms_column_bigger = True
        rescaled_ms = rescaled_ms[:, : img_pan.shape[1], :]
    else:
        ms_column_bigger = False
        img_pan = img_pan[:, : rescaled_ms.shape[1]]
  
    del img_ms; gc.collect()
  
  
    if ms_row_bigger == True and ms_column_bigger == True:
        img_psh = np.zeros((img_pan.shape[0], img_pan.shape[1], rescaled_ms.shape[2]), dtype = metadata_pan['dtype'])
    elif ms_row_bigger == False and ms_column_bigger == True:
        img_psh = np.zeros((rescaled_ms.shape[0], img_pan.shape[1], rescaled_ms.shape[2]), dtype = metadata_pan['dtype'])
        metadata_pan['height'] = rescaled_ms.shape[0]
    elif ms_row_bigger == True and ms_column_bigger == False:
        img_psh = np.zeros((img_pan.shape[0], rescaled_ms.shape[1], rescaled_ms.shape[2]), dtype = metadata_pan['dtype'])
        metadata_pan['width'] = rescaled_ms.shape[1]
    else:
        img_psh = np.zeros((rescaled_ms.shape), dtype = metadata_pan['dtype'])
        metadata_pan['height'] = rescaled_ms.shape[0]
        metadata_pan['width'] = rescaled_ms.shape[1]
        
    

    
    if method == 'simple_brovey':
        all_in = rescaled_ms[:, :, R - 1] + rescaled_ms[:, :, G - 1] + rescaled_ms[:, :, B - 1] + rescaled_ms[:, :, NIR - 1]
        for band in range(rescaled_ms.shape[2]):
            print(band)
            img_psh[:, :, band] = np.multiply(rescaled_ms[:, :, band], (img_pan / all_in))
        
  
    if method == 'simple_mean':
        for band in range(rescaled_ms.shape[2]):
            img_psh[:, :, band] = 0.5 * (rescaled_ms[:, :, band] + img_pan)
    
        
    if method == 'esri':
        ADJ = img_pan - rescaled_ms.mean(axis = 2)
        for band in range(rescaled_ms.shape[2]):
            img_psh[:, :, band] = rescaled_ms[:, :, band] + ADJ
        
    
    if method == 'brovey':
        DNF = (img_pan - W * rescaled_ms[:, :, NIR - 1]) / (W * rescaled_ms[:, :, R - 1] + W * rescaled_ms[:, :, G - 1] + W * rescaled_ms[:, :, B - 1])
        for band in range(rescaled_ms.shape[2]):
            img_psh[:, :, band] = rescaled_ms[:, :, band] * DNF
        
  
    del img_pan, rescaled_ms; gc.collect()
  
    
    # metadata_pan['count'] = img_psh.shape[2]
    # with rasterio.open(psh, 'w', **metadata_pan) as dst:
    #     dst.write(np.transpose(img_psh, [2, 0, 1]))
    
    io.export_geoTIFF(psh, img_psh, metadata_pan)
    return img_psh
            
            
            
class tilesGenerator():
    def __init__(self, image, stepSize, x_offset = 0, y_offset = 0):
        assert utils.is_channel_last(image)
        #parameters defined by user
        self.im = image
        self.T = stepSize
        self.x_offset = x_offset
        self.y_offset = y_offset
        
        #internal parameters       
        self.n_tiles_per_row = int(np.floor( (self.im.shape[0] - self.x_offset) /self.T))
        self.n_tiles_per_col = int(np.floor( (self.im.shape[1] - self.y_offset) /self.T))        
        self.indeces = (np.zeros((2, self.n_tiles_per_row * self.n_tiles_per_col)) +1).astype(int)
        
        #---Storing the tiles is too much memory-consuming---
        # self.tiles = np.empty((self.n_tiles_per_row * self.n_tiles_per_col, self.im.shape[0], self.T, self.T)).astype('float32')
        self.tiles = np.empty((self.n_tiles_per_row * self.n_tiles_per_col, self.T, self.T, self.im.shape[-1])).astype(image.dtype)

        
    def image2tiles(self):
        #Create an array of tiles and an array with indeces of each tile in the initial big image
        k = 0  
        for row in range(self.x_offset, self.T*self.n_tiles_per_row, self.T):
            for col in range(self.y_offset, self.T*self.n_tiles_per_col, self.T):
                self.indeces[0, k] = row
                self.indeces[1, k] = col
                
                window = self.im[row:row+self.T, col:col+self.T, :]                
                self.tiles[k,:,:,:] = window
                k = k+1
                
    def tiles2image(self, tiles_pred):
        map_pred = np.empty( (self.im.shape[0], self.im.shape[1]) ) #it is only one channel
        
        for k in range(0, len(tiles_pred)):
            row = self.indeces[0, k]
            col = self.indeces[1, k]
            map_pred[row:row+self.T, col:col+self.T] = tiles_pred[k] 
        return map_pred


def extract_extent(image_src, save = False):
    bounding_box = image_src.bounds
    polygon = box(*bounding_box)
    vector_file = geopandas.GeoDataFrame(index=[0], geometry = [polygon], crs = image_src.crs)
    if save:
        vector_file.to_file("polygon_extent.json", driver = "GeoJSON")
    image_src.close()
    return vector_file



## Supersed by 'split_satellite_with_AOI'
def clip_satellite_with_AOI(satellite_image_file : str , AOI_file : geopandas.GeoDataFrame()):
    with rasterio.open(satellite_image_file) as src:
        # Clip the GeoTIFF
        # Convert the AOI geometries to the format expected by rasterio.mask.mask
        aoi_shapes = [feature["geometry"] for _, feature in AOI_file.iterrows()]
        clipped, clipped_transform = mask(src, aoi_shapes, crop=True)
        
        # Update the metadata for the clipped raster
        clipped_meta_data = src.meta.copy()
        clipped_meta_data.update({
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform
        })
    return clipped, clipped_meta_data 


def split_satellite_with_AOI(satellite_image_file : str, AOI : geopandas.GeoDataFrame, 
                             tmp_folder = "tmp", verbose = False):
    """
    Clip a GeoTIFF image into multiple smaller GeoTIFFs, each corresponding to 
    a polygon from a GeoPandas GeoDataFrame. The polygons are separated.
    """
    # Create a tmp folder to store the small images    
    if not os.path.exists(tmp_folder):
        print("--creationg temporary folder")
        os.makedirs(tmp_folder)
        
    print("--split image from AOIs into {} parts".format(len(AOI)))
    with rasterio.open(satellite_image_file) as src:
        # Iterate over the polygons in the GeoDataFrame
        for index, row in AOI.iterrows():
            # Get the polygon geometry in a format that rasterio accepts
            geom = [mapping(row['geometry'])]
            # Perform the clip
            out_image, out_transform = mask(src, geom, crop=True)
            out_meta = src.meta.copy()

            # Update the metadata to reflect the number of layers,
            # and the new transformation matrix
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            # Path for the output GeoTIFF with the clip corresponding to the current polygon
            output_geotiff = f'{index}.tif' # Naming each output file uniquely

            # Write the clipped area to a new GeoTIFF
            io.export_geoTIFF(tmp_folder + os.sep + output_geotiff, out_image, out_meta, verbose = verbose)




def merge_geotiffs(input_files, output_file):
    # Open all the input files
    src_files_to_mosaic = []
    for f in input_files:
        src = gdal.Open(f)
        src_files_to_mosaic.append(src)
    
    # Create a virtual raster that contains all the input files
    vrt = gdal.BuildVRT("temporary.vrt", src_files_to_mosaic)
    # Create the output file based on the virtual raster
    gdal.Translate(output_file, vrt)
    
    # Clean up
    vrt = None
    for src in src_files_to_mosaic:
        src = None
       
    os.remove("temporary.vrt")







    
    
    