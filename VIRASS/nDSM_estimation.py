"""
@author: Michele Gazzea
"""

from . import RSClassifier
from . import architectures
from . import stitching
from . import utils
from . import geo_utils

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
        
        
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




class nDSMmodeler(RSClassifier.RSClassifier):
    
    def __init__(self, config_file: str = 'config.yml'):
        super().__init__()
        if not os.path.exists(config_file):
            config_file = utils.open_file("Open configuration file (yaml format)")
            
        if config_file.split('.')[1] == 'yaml':
            self.config = utils.read_config(config_file)
        else:
            raise ValueError("Wrong format for the configuration file")

    
    def instantiate_model(self, input_shape, architecture):
        model_path = self.working_dir + self.config['config_training']['model_name'] +'.h5'
        
        if os.path.isfile(model_path): 
            print("--Model already exist. Call <load_model> class method")
        else:  
            
            if architecture == "unet":                                                                
                model = architectures.unet(input_shape = input_shape, task = "regression")
            elif architecture == "unet_attention":
                dropout = self.config['config_training']['dropout']
                model = architectures.UNet_Attention(input_shape = input_shape, task = "regression", dropout = dropout)
            else:
                raise NotImplementedError()
           
            opt = tf.keras.optimizers.Adam(learning_rate = self.config['config_training']['lr']) 
            loss = tf.keras.losses.MeanSquaredError()
            metrics = [tf.keras.metrics.MeanAbsoluteError()]
            model.compile(loss = loss, 
                        optimizer = opt, 
                        metrics = metrics)            
            print("Model initialized\n")
            # model.summary()
            self.DLmodel = model
            
    
    def _pre_process_dataset(self, X, y):
        X = (X / 255.0).astype('float32')
        return X, y
    
        
    def build_model(self,):
        model_path = self.working_dir + self.config['config_training']['model_name'] +'.h5'
                
        if os.path.isfile(model_path):
            #if the model already exists, load it
            self.load_model()
        else:
            print("--Model does not exist. Creating a new one:")
            
            patch_number = self.config['config_training']['patch_number']
            patch_radius = self.config['config_training']['patch_radius']
            
            
            # Create training dataset
            X, y = self.create_dataset(patch_number = patch_number, 
                                       patch_radius = patch_radius, 
                                       save_patch = False, 
                                       export_patch_locations = False)
            
            # Instantiate and train model
            self.instantiate_model(input_shape = (2 * patch_radius, 2 * patch_radius, X.shape[-1]), 
                                   architecture = self.config['config_training']['architecture'])
            X, y = self._pre_process_dataset(X, y)
            
            print("--Training model on the following dataset: \n")
            print(X.shape, y.shape)
            self.train_model(X, y, model_path)
            
            
    def train_model(self, X, y, model_path = ""):  
        """Train model:
            X: patches extracting from the SAT images as (Npatches, radius, radius, Nchannels)
            y: patches extracting from the Ground truth as (Npatches, radius, radius, Nchannels = 1)"""
             
        if os.path.isfile(model_path):
            #if the model already exist, don't do anything
            print("--Model already exists. Call <load_model> class method")
        else:
            print("--Train model")               
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            #Train
            my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose = 1)] #10
            
            history = self.DLmodel.fit(X_train, y_train, 
                          epochs = self.config['config_training']['n_epochs'], 
                          batch_size = 128,
                          validation_data = (X_test, y_test),
                          callbacks = my_callbacks)
            
            # Export model and training history
            if not os.path.isfile(model_path): 
                self.DLmodel.save(model_path)
            with open(model_path.split('.')[0] + "_history", 'wb') as file:
                pickle.dump(history.history, file)
                
                
    def load_model(self):
        model_path = self.working_dir + self.config['config_training']['model_name'] +'.h5'
        if os.path.isfile(model_path):
            self.DLmodel = tf.keras.models.load_model(model_path, compile = False)
            print("--Model loaded")
        else:
            print("--Model does not exist")




    def generate_nDSM(self, image):
        if self.DLmodel is not None:
            # How much memory would the tiles consume roughly? in GB
            nb_classes = 1
            window_size = 2 * self.config['config_training']['patch_radius']
            subdivisions = 2
            N_tiles_rows = subdivisions * image.shape[0] / window_size
            N_tiles_cols = subdivisions * image.shape[1] / window_size
            # 4 is the factor from uint8 to float32 reppresentation, 1024**3 is the byte->GB conversion
            ESTIMATED_MEMORY = (N_tiles_rows * N_tiles_cols * window_size**2 * image.shape[2] * 4 * nb_classes)/(1024**3)
            print("Estimated memory {} GB \n". format(ESTIMATED_MEMORY))
            
            # if the data require less than 5 GB, then fit it all
            if ESTIMATED_MEMORY < 5:
                # tree mask can be computed a straight way
                stiched_pred_map = self._generate_nDSM(image,  window_size, subdivisions)
            else:
                
                if utils.confirmCommand("Image too large, split it?"):
                    # otherwise if the input image is too big, need to split it
                    # how many chunk? Each chunk is ~4GB, then we need N chunks
                    N = int(np.ceil(ESTIMATED_MEMORY / 4))
                    print("Satellite image is split into {} parts".format(N))
                    # Divide the image into N parts: each is saved temporarly (tmp folder is created). 
                    tmp_folder = "tmp"
                    geo_utils.split_raster_to_parts(image, N, tmp_folder)
                    
                    stiched_pred_map = []
                    # For each chunk
                    for i in range(0, N):
                        # Load the chunk
                        print("Load part: " + str(i))
                        part = np.load(tmp_folder + "/part" + str(i) + ".npy")
                        # Segment it
                        print("Segment part: " + str(i))
                        stiched_pred_map.append(self._generate_nDSM(part, window_size, subdivisions))
                        # Save the mask
                        print("Save predicted: " + str(i))
                        
                    # When all mask are created, merge them back together
                    stiched_pred_map = np.vstack(stiched_pred_map)
                    
                    # Delete the input image
                    # TODO
                else:
                    stiched_pred_map = None
                                  
            return stiched_pred_map
        
        else:
            print("--Model not loaded")


    def _generate_nDSM(self, image, window_size, subdivisions):                    
            image = (image / 255).astype('float32')
            print("Segmenting trees with tiles smoothing function")

            stiched_pred_map = stitching.predict_img_with_smooth_windowing(image, window_size, subdivisions, nb_classes=1, 
                                          pred_func = self.DLmodel.predict)
                
            # stiched_pred_map = (stiched_pred_map*255).astype('uint8')
            return stiched_pred_map

    

    def plot_training_history(self, ):
        
        with open(self.working_dir + self.config['config_training']['model_name'] + "_history", "rb") as input_file:
            history = pickle.load(input_file)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.plot(history['loss'], label = "MSE (training)")
        plt.plot(history['val_loss'], label = "MSE (validation)")
        
        ax1.plot(history['mean_absolute_error'], label = "MAE (training)")
        plt.plot(history['val_mean_absolute_error'], label = "MAE (validation)")
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metrics')
        ax1.set_ylim(0, 20)
        
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(history['lr'], label = "learning rate", color='purple', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='purple')
        # ax2.set_ylabel('Learning rate')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        plt.legend(lines, labels, loc = 'best') #prop={'size': 20}
        plt.tight_layout()
        
        
        
        
    


    
def compute_performance(GT, pred):
        
    GT, pred = geo_utils.align_maps(GT, pred)
    
    GT = GT.ravel()
    pred = pred.ravel()
    
    # --Calculate evaluation metrics
    
    # Mean Absolute Error
    mae = mean_absolute_error(GT, pred)  
    print("\nMean Absolute Error (MAE): {:.2f}".format(mae))
    
    # Mean Squared Error
    mse = mean_squared_error(GT, pred)        
    # Root Mean Squared Error
    rmse = np.sqrt(mse)                           
    print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
    
    # R-squared
    r2 = r2_score(GT, pred)              
    print("R-squared (R2): {:.2f}".format(r2))
    
    mask = np.where(GT != 0)
    error = (GT[mask] - pred[mask]).ravel()
    plt.hist(abs(error), bins = 50)
    plt.xlabel("Error [meters]")
    plt.ylabel("distribution")
    # plt.yscale("log")
    plt.tight_layout()
    
    
    
    
    
























