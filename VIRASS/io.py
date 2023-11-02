"""
Input-Output functions
"""

import numpy as np
import os
import rasterio
import geopandas

from . import utils


def open_geoTiFF(filename, with_meta_data = True):
    """
    Parameters
    ----------
    filename : string
        path of the geoTiFF image.
    with_meta_data : bool, optional
        Whether to acquire the meta data of the geTiFF file or not. The default is True.

    Returns
    -------
    np.array
        Return the image as a numpy array. It also return the meta data if 'with_meta_data' is True.

    """
    
    if not os.path.isfile(filename):
        relocate = utils.confirmCommand("File: <"+ filename + "> does not exist. Do you want to locate it in other directories?")
        if relocate:
            # ask for new filename
            filename = utils.open_file(title = "Open GeoTIFF image")
        else: 
            filename = None
            print("geoTiFF not loaded")

    if filename is not None:
        # open GeoTIFF file and extract the image
        file_src = rasterio.open(filename)
        meta_data = file_src.profile    
        bands = range(1, meta_data['count']+1)
        SAT_image = file_src.read(bands)
        file_src.close()
        
        # transpose the image as channel-last
        SAT_image = utils.convert_to_channel_last(SAT_image)
        if with_meta_data:
            return SAT_image, meta_data
        else:
            return SAT_image
    
    
    
    
def export_GEOtiff(filename: str, SAT_image: np.array, meta_data: dict):
    """
    Export a numpy array "Map" into GeoTIFF
    Input:  filename: path+name.tif of the exported output  
            SAT_image: numpy array
            meta_data: GeoTIFF meta data to include in the output
    """ 
    if SAT_image is not None:
        meta_data['dtype'] = SAT_image.dtype
        if len(SAT_image.shape)==2:
            # add the depth dimension
            SAT_image = np.expand_dims(SAT_image, axis=-1)
        
        SAT_image = utils.convert_to_channel_first(SAT_image)
        meta_data['count'] = SAT_image.shape[0]    
        assert meta_data['count'] <=15 #sanity check: bands should be less than 8 in normal images
        
        if os.path.isfile(filename):
            if utils.confirmCommand("The file "+ filename + " already exist. Do you want to overwrite it?"):
                with rasterio.open(filename, 'w', **meta_data) as dst:
                    dst.write(SAT_image)
        else:
            with rasterio.open(filename, 'w', **meta_data) as dst:
                dst.write(SAT_image)
            
            
            
def load_infrastructure_line(filename: str, crs = 32632):
    #if file does not exist
    if not os.path.isfile(filename):
        relocate = utils.confirmCommand("File: <"+ filename + "> does not exist. \n Do you want to locate it in other directories?")
        if relocate:
            # ask for new filename
            filename = utils.open_file(title = "Open infrasctructure file")
        else: 
            filename = None
        
    if filename is None:
        print("Infrastructure file not loaded")
        return None
    else:
        # Load and convert to a suitable CRS
        return geopandas.read_file(filename).to_crs("epsg:" + str(crs))
            
            
            
            
            
            
            
            
            
            