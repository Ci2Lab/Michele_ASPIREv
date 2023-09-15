"""
@author: Michele Gazzea
"""

import numpy as np
import os
import PIL

import shapely
import geopandas as gpd
import pandas as pd
import json

from . import utils
from . import io
from . import geo_utils


class RSClassifier():    
    """ Basic Remote Sensing (RS) Classifier.
    
    It stores the X_map (satellite input) and possibly a y_map (reference or ground truth).
    The class is inherited by the other classes to perform more specific tasks (tree segmentation, 3d modeling, ...) 
    """
    
    # TODO: refactor the attributes creation in a way that attributes and class behaviour is chosen
    #  from a YAML configuration file. 
    #  Similar to DeepForest: https://github.com/weecology/DeepForest/blob/main/deepforest/main.py
       
    
    def __init__(self):
        
        # Basic attributes
        self.X_map = None
        self.y_map = None
        self.meta_data = None
        self.DLmodel = None
        self.patch_radius = 0
        
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
            
        print("\n***class <" + self.__class__.__name__ + "> initialized ***\n")
         
        
        # Specify a working direcotry to store data
        self.working_dir = utils.open_dir("Specify a directory to save data:")
        if self.working_dir == "":
            self.working_dir = os.path.abspath(os.getcwd())
            
        if self.working_dir[-1] != '\\':
            self.working_dir = self.working_dir + '\\' 
            
            
        #     self.config['working_dir'] = os.path.abspath(os.getcwd())
        # # if working directory is not defined, ask for it:
        # if not "working_dir" in self.config:
        #     print("Key <working_dir folder> in config is not defined.")                   
        #     self.config['working_dir'] = utils.open_dir("Specify a directory to save data:")        
        # # if empty, set it as the current directory 
        # if self.config['working_dir'] == "":
        #     self.config['working_dir'] = os.path.abspath(os.getcwd())              
        # print("<working_dir folder> set to: " + self.config['working_dir'])
            
        # if self.config['working_dir'][-1] != '\\':
        #     self.config['working_dir'] = self.config['working_dir'] + '\\' 
    
    
    def print_attributes(self):
        """ Print class attrbutes """
        # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
        print("\n---Printing class attributes:---")
        for attribute, value in self.__dict__.items():
            if hasattr(value, "shape"):
                print(f"{attribute} = {value.shape}")
            else:
                print(f"{attribute} = {value}")
        print("\n")
        
        
   
           
    # %% Check functions
    
    def check_X_map(self):
        if self.X_map is None:
            print("X_map not loaded")
            if utils.confirmCommand("Do you want to locate it?"):
                self.X_map = self.load_X_map()
    
    def check_y_map(self):
        if self.y_map is None:
            print("y_map not loaded")
            if utils.confirmCommand("Do you want to locate it?"):
                self.y_map = self.load_y_map()
    
    def check_Xy_maps(self):
        """ Check if X-map and y_map are loaded and if they have the same dimensions.
        If not (most of the time they differ by 1 pixel because of reprojections), then align them
        """
        # Check X_map and y_map
        self.check_X_map()            
        self.check_y_map()
            
        if (self.X_map is not None and self.y_map is not None):   
            # Check dimensions        
            print("X_map shape {} \ny_map shape {}".format(self.X_map.shape, self.y_map.shape))
            if not (self.X_map.shape[0:2] == self.y_map.shape[0:2]):
                # Align dimensions
                self.align_X_y()
  
        
    # %% IO functions
    
    @staticmethod
    def open_file(file_path = None, name = "file_name"):
        if (file_path is None) or (not os.path.isfile(file_path)):
            # if file_path not provided or not exist:: ask manually for it
            file_path = utils.open_file(title = "Locate " + name)
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".tif":
            # If it is a GeoTIFF, open it as it:
            _map, meta_data = io.open_geoTiFF(file_path)
        else:
            _map = np.array(PIL.Image.open(file_path))
            print("WARNING! - meta data not found")
            meta_data = None
        return _map, meta_data
    
    
    def load_X_map(self, X_map_filepath = None, convert_to_RGB = False, source = ""):
        print("\n Loading X_map")
        X_map, meta_data = self.open_file(X_map_filepath, "X_map")
        
        if convert_to_RGB:
            X_map = geo_utils.multiband_to_RGB(X_map, source) 
        print("X_map shape:" + str(X_map.shape))
        self.X_map = X_map
        self.meta_data = meta_data
        print("X_map loaded \n")
        
        
    def load_y_map(self, y_map_filepath = None):  
        print("\n Loading y_map")
        y_map, meta_data = self.open_file(y_map_filepath, "y_map")
        assert len(y_map.shape) == 3
        
        if y_map.shape[-1] > 1:
            print("-- selecting only the first component")
            y_map = y_map[:,:,0:1]
        
        print("Ground truth shape:" + str(y_map.shape))
        self.y_map = y_map
        print("y_map loaded \n")
        
        
    def save_config(self):
        with open(self.name + 'config.json', 'w') as fp:
            json.dump(self.working_dir, fp)
    
    
   
    # %% Processing functions
    
    def align_X_y(self):
        """Height/width between X_map and y_map might be off by one pixel due to some clipping.
        This will perform a small alignment"""
        if (self.X_map is None) or (self.y_map is None):
            print("X_map or y_map not loaded")
        else:
            if self.X_map.shape == self.y_map.shape:
                print("X_map and y_map have the same dimensions")
            else:
                print("X_map shape {} \ny_map shape {}".format(self.X_map.shape, self.y_map.shape))
                print('--Resizing to same shape')
                h, w, _ = [min(s) for s in zip(self.X_map.shape,  self.y_map.shape)]
                self.X_map = self.X_map[:h,:w,:]
                self.y_map = self.y_map[:h,:w,:]
                print("SAT shape {} \ny_map shape {}".format(self.X_map.shape, self.y_map.shape))
    
    
    
    #@f.measure_time
    def create_dataset(self,  patch_number = 10, patch_radius = 40, save_patch = False, export_patch_locations = False):
          
        #Internal method
        def _create_dataset(self, patch_number, patch_radius, save_patch, export_patch_locations):
            """Extract random patches for training from <X_map> (SAT image) and <y_map> (ground truth).
            X_map should be shaped as (heigth, width, Nchannels) 
            y_map should be shaped as (heigth, width, Nchannels = 1) 
            """
            
            #Requirement is that X_map and y_map are not None
            if (self.X_map is not None and self.y_map is not None):
                    
                radius = patch_radius
                N_patches = patch_number                
                
                # The list implementation is faster than index assign a value within a for loop 
                X = []
                Y = []
                 
                assert self.X_map.shape[0:2] == self.y_map.shape[0:2]  
                height = self.X_map.shape[0]
                width = self.X_map.shape[1]
                #select random row and col
                np.random.seed(0) # make it predictable
                rows = np.random.randint(low = radius, high = height-radius-1, size = (N_patches,))
                cols = np.random.randint(low = radius, high = width-radius-1, size = (N_patches,))
                
                for n in range(0, N_patches):
                    row = rows[n]; col = cols[n]
                    X.append(self.X_map[row-radius:row+radius, col-radius:col+radius,:])
                    Y.append(self.y_map[row-radius:row+radius, col-radius:col+radius, :])
                
                X = np.asarray(X)
                Y = np.asarray(Y)
                
                if save_patch:
                    np.save(self._config['saving_folder'] + "X_patches", X)
                    np.save(self._config['saving_folder'] + "y_patches", Y)
                    
                if export_patch_locations:
                    # translate rows and cols value into geographical locations
                    if self.meta_data is not None:
                        pointX = cols * self.meta_data['transform'][0] + self.meta_data['transform'][2]
                        pointY = rows * self.meta_data['transform'][4] + self.meta_data['transform'][5]
                        pointXY = (pointX, pointY)                                                                     
                        df1 = pd.DataFrame({'pointX':pointXY[0], 'pointY':pointXY[1]})
                        df1['geometry'] = list(zip(df1['pointX'], df1['pointY']))
                        df1['geometry'] = df1['geometry'].apply(shapely.geometry.Point)
                        gdf1 = gpd.GeoDataFrame(df1, geometry='geometry', crs="EPSG:32632") # 32632;  25832
                        # Create Polygon geometry to show the patches as rectangles
                        gdf1['geometry'] = gdf1.geometry.buffer( int(radius*self.meta_data['transform'][0]), cap_style = 3)
                        gdf1.to_file(self.config['working_folder'] + "patches_locations.shp")
                    
                    else:
                        print("Metadata is not defined. Impossible to locate patches.")
                        
                            
                    
                return X, Y
            
            else:
                raise RuntimeError("X_map not loaded. Couldn't create a dataset")
                
                
        """start"""
        self.check_Xy_maps() 
        # TODO: Eventually handle the exception with try-catch 
        X, Y = _create_dataset(self, patch_number = patch_number,
                                    patch_radius = patch_radius,
                                    save_patch = save_patch,
                                    export_patch_locations = export_patch_locations)
        return X, Y
            




















