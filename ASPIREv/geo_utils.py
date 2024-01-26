"""
@author: Michele Gazzea
"""
import matplotlib.pyplot as plt
from osgeo import ogr, osr
import numpy as np
from scipy import ndimage
from skimage import morphology 
import os
import rasterio
from affine import Affine
from shapely.geometry import shape
import geopandas
import cv2
import gc


from . import utils
from . import io
  
 
def multiband_to_RGB(SAT_image, source = "WorldView"):
    """
    Take the RGB components of a satellite multi-channel image
    """
    
    # Ensure the bands are in the last axis
    assert utils.is_channel_last(SAT_image)
    
    if source == "WorldView":
        assert SAT_image.shape[-1] == 8
        SAT_image = np.take(SAT_image, indices = [4,2,1], axis = -1)
    elif source == "Pleiades":
        assert SAT_image.shape[-1] == 4
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
    
    
  
    
def preProcessImage(SAT_filename, save=False, NDVI = True, shift = False):
    """
    Take the 4Bands uint16 raw image and process it to convert it into uint8 image [0-255].
    Cumulative count cut to 2-98% percentile and MinMax stretch is applied
    Preprocessing consist of:
    - color balancing
    - shifting (in case)
    - NDVI calculation (in case)
    ---
    Input: path to the SAT image
    Output: preprocessed image  [uint8]
    """
    
    def shiftImage(meta_data, Xoffset, Yoffset):
        meta_dataNew = meta_data#.copy() #it keeps the old one
        meta_dataT = meta_dataNew['transform']
        meta_dataNew['transform'] = Affine(meta_dataT[0], meta_dataT[1], meta_dataT[2]+Xoffset , \
                               meta_dataT[3], meta_dataT[4], meta_dataT[5]+Yoffset)
        return meta_dataNew

    print("--Preprocessing image: " + SAT_filename)
    #--Open GeoTIFF file and extract the image
    file_src = rasterio.open(SAT_filename)
    meta_data = file_src.profile    
    bands = range(1, meta_data['count']+1)
    Map = file_src.read(bands).astype('float32')
    file_src.close()
    
    if shift:
        print("--Shift image")
        # meta_data = shiftImage(meta_data, 4, 0) #Shift image: Pleiades 
        meta_data = shiftImage(meta_data, 8.41, 1.5) #Shift image: WorldView
    
    if NDVI:        
        print("--Calculating NDVI")
        # --Compute the NDVI
        if meta_data['count']==4:
            NDVI_map = np.expand_dims( ( Map[3,:,:] - Map[0,:,:] ) / ( Map[3,:,:] + Map[0,:,:] ), axis=0)
        elif meta_data['count']==8:
            NDVI_map = np.expand_dims( ( Map[6,:,:] - Map[4,:,:] ) / ( Map[6,:,:] + Map[4,:,:] ), axis=0)
        else:
            NDVI_map = None
        
        NDVI_map[np.where(NDVI_map < -1 )] = np.nan
        a = np.nanpercentile(NDVI_map.flatten(), 2)
        b = np.nanpercentile(NDVI_map.flatten(), 98)
        NDVI_map = np.clip( ( ( (NDVI_map - a) / (b-a) ) * 255) , 0,255).astype('uint8')
        
    #--Process the raw uint16 satellite bands to enhance contrast (=color balance version)
    alpha = np.expand_dims( np.clip( np.sum(Map, axis=0), 0, 1), axis=0 )
    alpha[np.where(alpha==0)] = np.nan
    
    Map = alpha * Map
        
    for i in range(0, meta_data['count']):
        print("--Processing band: " + str(i+1))
        
        a = np.nanpercentile(Map[i,:,:].flatten(), 2)
        b = np.nanpercentile(Map[i,:,:].flatten(), 98)
        Map[i,:,:] = np.clip( ( ( (Map[i,:,:] - a) / (b-a) ) * 255) , 0,255)    
    
        Map[np.where(Map[i,:,:] == np.nan)] = 0
    output = Map.astype('uint8')
    
    if NDVI:
        #--Concatenate RGB with NDVI 
        output = np.concatenate( (output, NDVI_map), axis=0 ) 
        

    #--Export it as GeoTIFF file    
    if save:
        print("--Saving image")
        meta_data['nodata'] = None
        filename = SAT_filename.split('.')[-2] #remove ".tif" form the filename
        io.export_GEOtiff(filename+'_preProcessed.tif', output, meta_data)    
        print("--Image saved in: " + filename + '_preProcessed.tif')
    return output    



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



def create_alpha_channel(SAT_image, meta_data):
    assert utils.is_channel_last(SAT_image)
    """ no data have 0 in all channels. This works most of the time but some pixels have 0 values in all channels
    even if they are valid pixels, need to distiguish the outer layer from the region within the satellite """    
            
    alpha_layer = np.clip(np.sum(SAT_image, axis = -1), 0, 1)
    
    # Perform morphological closing on the pre-computed alpha channel
    alpha_layer = (np.expand_dims( morphology.binary_closing(alpha_layer, 
                                  selem = morphology.disk(radius = 3)), 
        axis=-1)*255).astype('uint8')
    return alpha_layer



def raster_to_vector(input_raster = "alpha_channel.tif", output_vector = "output_vector.gpkg"):
    crs = 'epsg:32632' 
    with rasterio.open(input_raster) as src:
        mask = src.read(1)
        shapes = list(rasterio.features.shapes(mask, transform=src.transform))

    # Filter shapes based on the white pixels
    white_shapes = [s for s in shapes if s[1] == 255]
    shape_info = white_shapes[0]
    polygon = shape(shape_info[0])
    gdf = geopandas.GeoDataFrame(crs = crs, geometry = [polygon])
    gdf.to_file(output_vector, driver="GPKG")




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
    
    
def split_raster_to_parts(SAT_image, N, tmp_folder = "tmp"):
    
    # Create a temporary folder to store the parts of the image
    
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    assert utils.is_channel_last(SAT_image)
    
    if len(os.listdir(tmp_folder)) == 0:        
        # Here we split the image by rows, doing it by columns is the same
        rows_per_part = np.ceil(SAT_image.shape[0] / N)
        for i in range(0, N):
            a = int(i*rows_per_part)
            b = int((i+1)*rows_per_part)
            part = SAT_image[a:b,:,:] 
            np.save(tmp_folder + "/part" + str(i), part)
            # Image.fromarray(part).save(new_folder + "/part" + str(i) + ".jpeg")
    else:
        print("Parts already exist")


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
    
    io.export_GEOtiff(psh, img_psh, metadata_pan)
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

















    
    
    