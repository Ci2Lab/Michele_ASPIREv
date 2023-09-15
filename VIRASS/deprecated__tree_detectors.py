"""
Tree detector class. It implements three different models:
    1-the 3D generation model
    2-tree segmenter
    3-tree species classifier
    
Tree crown delineation is performed based on the outputs of the models above.
"""
import utils_functions as f
import utils_DL_architectures
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import shapely
import geopandas as gpd
import pandas as pd
import warnings
from scipy import ndimage
import os
import PIL
from sklearn.model_selection import train_test_split

def extract_tree_mask(nDSM_path, NDVI_path, height_threshold=1.3, ndvi_threshold=0.1):
    """
    Extract trees from nDSM (-> height threshold) and NDVI (-> vegetation index threshold)
    """
    # Fetch nDSM and NDVI
    nDSM, meta_data = f.open_GeoTiFF(nDSM_path)
    NDVI = f.open_GeoTiFF(NDVI_path, with_meta_data=False)
    
    # during preprocessing height/width might be off by one pixel
    if nDSM.shape != NDVI.shape:
        print("nDSM shape {} \nNDVI shape {}".format(nDSM.shape, NDVI.shape))
        print('--Resizing to same shape')
        _, h, w = [min(s) for s in zip(nDSM.shape, NDVI.shape)]
        nDSM = nDSM[:,:h,:w]
        NDVI = NDVI[:,:h,:w]
        
    
    # tree points: height enough and green enough
    tree_height = nDSM > height_threshold
    green = NDVI > ndvi_threshold
    tree_points = np.squeeze(np.logical_and(tree_height, green).astype('uint8'), axis=0)
    # remove isolated points and noise
    # return tree_points, meta_data
    
    def erode_and_dilate(source, structure=np.ones((2,2)), it=1):
        eroded = ndimage.binary_erosion(source, structure=structure, iterations=it).astype(source.dtype)
        dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=it*2).astype(eroded.dtype)
        result = np.logical_and(source,dilated).astype(dilated.dtype)
        return result

    tree_points = erode_and_dilate(tree_points, structure=np.ones((2,2)), it=2)
    return tree_points, meta_data 



class nDSM_generator():
    """
    Class that implements the deep learning model for nDSM generation from satellite image.
    It is trained with a LiDAR-based nDSM.
    """
    def __init__(self, config = {}):        
        self.SAT = None
        self.nDSM = None
        self.meta_data_SAT = None
        self._config = config
        print("--New nDSM initialized--")
        
    @staticmethod
    def open_file(file_path = None, name = "file_name"):
        if (file_path is None) or (not os.path.isfile(file_path)):
            # if file_path not provided or not exist:: ask manually for it
            file_path = f.open_file(title = "Locate" + name)
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".tif":
            # If it is a GeoTIFF, open it as it:
            _map, meta_data = f.open_GeoTiFF(file_path)
        else:
            _map = np.array(PIL.Image.open(file_path))
            print("meta data not found")
            meta_data = None
        return _map, meta_data
            
    
    def load_nDSM(self, y_map_filepath):  
        print("\n Loading nDSM")
        y_map, _ = self.open_file(y_map_filepath, "nDSM")
        assert len(y_map.shape) == 3
        #Chech if y_map is shaped correctly
        
        #Chech if y_map is shaped correctly
        if y_map.shape[0] == 1: #it means it is (Nchannels=1, heigth, width) 
            y_map = np.transpose(y_map, (1,2,0))
        else:
            pass
        
        print("Ground truth shape:" + str(y_map.shape))
        self.nDSM = y_map
        print("SAT loaded")
   
    
    def load_SAT(self, X_map_filepath = None):
        print("\n Loading SAT")
        X_map, meta_data = self.open_file(X_map_filepath, "SAT")
        #Chech if X_map is shaped correctly
        if X_map.shape[0] < 10: #it means it is shaped as: (Nchannels, heigth, width) 
            X_map = np.transpose(X_map, (1,2,0))
        else:
            pass
        
        print("SAT image shape:" + str(X_map.shape))
        self.SAT = X_map
        self.meta_data_SAT = meta_data
        print("SAT loaded")
    
        
    def check_status(self):
        """Check the status of the generator"""
        print("------- \n Checking status of the generator \n-------\n")
        # Check SAT and nDSM
        if (self.SAT is None) or (self.nDSM is None):
            warnings.warn("SAT or nDSM is not loaded \n")
        else:
            print("OK: SAT or nDSM are loaded \n")
            
            # Check dimensions        
            print("SAT shape {} \nnDSM shape {}".format(self.SAT.shape, self.nDSM.shape))
            if self.SAT.shape[:2] == self.nDSM.shape[:2]:
                print("OK: SAT or nDSM have same dimensions \n")
            else:
                print("SAT or nDSM have different dimensions \n")
    
    
    def align_X_y(self):
        """Height/width between SAT and nDSM might be off by one pixel due to some clipping.
        This will perform a small alignment"""
        if (self.SAT is None) or (self.nDSM is None):
            print("SAT or nDSM not loaded")
        else:
            if self.SAT.shape == self.nDSM.shape:
                print("SAT and nDSM have the same dimensions")
            else:
                print("SAT shape {} \nnDSM shape {}".format(self.SAT.shape, self.nDSM.shape))
                print('--Resizing to same shape')
                h, w, _ = [min(s) for s in zip(self.SAT.shape,  self.nDSM.shape)]
                self.SAT = self.SAT[:h,:w,:]
                self.nDSM = self.nDSM[:h,:w,:]
                print("SAT shape {} \nnDSM shape {}".format(self.SAT.shape, self.nDSM.shape))
            
            
    def create_dataset(self, patch_radius = 40, patch_number = 1000, save_patch = False):
        """Extract random patches for training from <X_map> (SAT image) and <y_map> (ground truth).
        X_map should be shaped as (heigth, width, Nchannels) 
        y_map should be shaped as (heigth, width, Nchannels = 1) 
        """        
        X_map = self.SAT
        y_map = self.nDSM
        
        radius = patch_radius
        N_patches = patch_number                
        Nchannels = X_map.shape[-1]
        X = np.empty((N_patches, 2*radius,2*radius, Nchannels)).astype(X_map.dtype)
        Y = np.empty((N_patches, 2*radius,2*radius, 1)).astype(X_map.dtype)
         
        assert (X_map.shape[0] == y_map.shape[0]) and (X_map.shape[1] == y_map.shape[1])  
        height = X_map.shape[0]
        width = X_map.shape[1]
        #select random row and col
        rows = np.random.randint(low = radius, high = height-radius-1, size = (N_patches,))
        cols = np.random.randint(low = radius, high = width-radius-1, size = (N_patches,))
        for n in range(0, N_patches):
            row = rows[n]; col = cols[n]
            X[n, :,:,:] = X_map[row-radius:row+radius, col-radius:col+radius,:]
            Y[n, :,:,:] = y_map[row-radius:row+radius, col-radius:col+radius, :]
            
        if save_patch:
            np.save(self._config['saving_folder'] + "X_patches", X)
            np.save(self._config['saving_folder'] + "y_patches", Y)
        
        return X, Y
       
    
    def detect_peaks(self,):
        if self.nDSM is None:
            warnings.warn("nDSM not loaded")
            return None
        else:
        # Detect trees by local maxima from nDSM
            return peak_local_max(np.squeeze(self.nDSM, axis=0), min_distance=6, threshold_abs=5)
                


def coords_to_geoCoords(coordinates, meta_data):    
    """
    transform coordinates (row, col) of image into geo-coordinates. 
    Output is a shapely Point shapefile.
    """ 
    pointX = coordinates[:,1] * meta_data['transform'][0] + meta_data['transform'][2]
    pointY = coordinates[:,0] * meta_data['transform'][4] + meta_data['transform'][5]
    
    return (pointX, pointY) 


def geoCoords_to_dataFrame(coordinates, meta_data):
    pointXY = coords_to_geoCoords(coordinates, meta_data)
    df1 = pd.DataFrame({'pointX':pointXY[0], 'pointY':pointXY[1]})
    df1['geometry'] = list(zip(df1['pointX'], df1['pointY']))
    df1['geometry'] = df1['geometry'].apply(shapely.geometry.Point)
    gdf1 = gpd.GeoDataFrame(df1, geometry='geometry', crs="EPSG:25832")
    gdf1.to_file("tree_locations.shp")
    

"""---------------------------------------------------------------------------"""




# trees, meta_data = extract_tree_mask("warping/nDSM_warped.tif", "_data/NDVI.tif")
# f.export_GEOtiff("trees.tif", trees, meta_data)



# modeler = nDSM_generation_model()
# modeler.load_DSM("warping/nDSM_warped.tif")
# peaks = modeler.detect_peaks()
# geoCoords_to_dataFrame(peaks, modeler.meta_data)


if __name__ == "__main__":
    modeler = nDSM_generator()
    modeler.load_SAT("_data/WorldView_area.tif")
    modeler.load_nDSM("warping/nDSM_warped.tif")
    modeler.check_status()
    modeler.align_X_y()
    
    X, y = modeler.create_dataset(patch_number = 2000)
    
    X = (X/255).astype('float32')
    y = (y/255).astype('float32')
    
    
    
    """TODO"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    import tensorflow as tf
    import keras
    model = utils_DL_architectures.myResUNet()
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001) #optimizer
    model.compile(
                loss = 'mse', 
                optimizer = opt)
    my_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)] #10
    model.fit(X_train, y_train, epochs = 100, 
                          batch_size = 64, 
                          validation_data = (X_test, y_test), 
                          callbacks=my_callbacks )


    model.save("3Dmodeler.h5")




import utils_stitching_f

input_img,meta_data = f.open_GeoTiFF("_data/WorldView_area_small.tif")

input_img = (np.transpose(input_img, (1,2,0))/255).astype('float32')

stiched_pred_map = utils_stitching_f.predict_img_with_smooth_windowing(input_img, window_size=80, subdivisions=2, nb_classes=1, 
                                               pred_func=model.predict, split_in_batches=True)


# tileGen = f.tilesGenerator2(input_img, stepSize = 80, x_offset = 0, y_offset = 0)
# tileGen.image2tiles()
# tiles = np.transpose(tileGen.tiles, (0,2,3,1))/255.
# pred = model.predict(tiles)[:,:,:,0]
# map_pred = (tileGen.tiles2image(pred) * 255).astype('uint8')

f.export_GEOtiff("nDSM_predicted.tif", stiched_pred_map, meta_data)














        