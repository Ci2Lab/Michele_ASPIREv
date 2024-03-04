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
        

import shutil        
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import morphology 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, jaccard_score




class TreeSegmenter(RSClassifier.RSClassifier):
    
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
                model = architectures.unet(input_shape = input_shape, n_classes = 1)
            elif architecture == "unet_attention":
                dropout = self.config['config_training']['dropout']
                model = architectures.UNet_Attention(input_shape = input_shape, n_classes = 1,  dropout = dropout)
            else:
                raise NotImplementedError()
           
            opt = tf.keras.optimizers.Adam(learning_rate = self.config['config_training']['lr']) 
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), 
                       tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
            model.compile(loss = loss, 
                        optimizer = opt, 
                        metrics = metrics)            
            print("Model initialized\n")
            # model.summary()
            self.DLmodel = model
            
    
    def _pre_process_dataset(self, X, y):
        X = (X / 255.0).astype('float32')
        y = (y / 255).astype('uint8')
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




    def generate_tree_map(self, image, MEMORY_LIMIT = 5, verbose = True, tmp_folder = "tmp_image_chunks"):
        if self.DLmodel is not None:
            
            assert utils.is_channel_last(image)
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
            if ESTIMATED_MEMORY < MEMORY_LIMIT:
                # tree mask can be computed a straight way
                stiched_pred_map = self._generate_tree_map(image,  window_size, subdivisions, verbose = verbose)
            else:
                
                if utils.confirmCommand("Image too large, split it?"):
                    # otherwise if the input image is too big, need to split it
                    # how many chunk? Each chunk is ~4GB, then we need N chunks
                    N = int(np.ceil(ESTIMATED_MEMORY / MEMORY_LIMIT))
                    print("Satellite image is split into {} parts".format(N))
                    # Divide the image into N parts: each is saved temporarly (tmp folder is created).                     
                    geo_utils.split_raster_to_parts(image, N, tmp_folder)
                    
                    stiched_pred_map = []
                    # For each chunk
                    for i in range(0, N):
                        # Load the chunk
                        print("Load part: " + str(i))
                        part = np.load(tmp_folder + "/part" + str(i) + ".npy")
                        # Segment it
                        print("Segment part: " + str(i))
                        stiched_pred_map.append(self._generate_tree_map(part, window_size, subdivisions, verbose = verbose))
                        # Save the mask
                        print("Save predicted: " + str(i))
                           
                        
                    # When all mask are created, merge them back together
                    stiched_pred_map = np.vstack(stiched_pred_map)
                    
                    # Delete the input image
                    utils.delete_tmp_folder(tmp_folder)
                else:
                    stiched_pred_map = None
                                  
            return stiched_pred_map
        
        else:
            print("--Model not loaded")


    def _generate_tree_map(self, image, window_size, subdivisions, verbose = True):                    
            image = (image / 255).astype('float32')
            if verbose:
                print("--Segmenting trees over tiles (with smoothing function)")

            stiched_pred_map = stitching.predict_img_with_smooth_windowing(image, window_size, subdivisions, nb_classes=1, 
                                          pred_func = self.DLmodel.predict)
                
            stiched_pred_map = (stiched_pred_map*255).astype('uint8')
            return stiched_pred_map

    

    def plot_training_history(self, ):
        
        with open(self.working_dir + self.config['config_training']['model_name'] + "_history", "rb") as input_file:
            history = pickle.load(input_file)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.plot(history['loss'], label = "loss (training)")
        ax1.plot(history['val_loss'], label = "loss (validation)")
        
        ax1.plot(history['binary_accuracy'], label = "binary_accuracy (training)")
        ax1.plot(history['val_binary_accuracy'], label = "binary_accuracy (validation)")
        
        ax1.plot(history['binary_io_u'], label = "IoU (training)")
        ax1.plot(history['val_binary_io_u'], label = "IoU (validation)")
        
        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metrics')
        
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(history['lr'], label = "learning rate", color='purple', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='purple')
        # ax2.set_ylabel('Learning rate')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        plt.legend(lines, labels, loc = 'best')
        plt.tight_layout()


           
                    
        
def binarize_treeMap(map_pred, thresholds, closing = True):
    
    """
    From a threshold list 'thresholds' containing one threshold per output
    channel for comparison, the predictions are converted to a binary mask.
    """
    assert (map_pred.shape[-1] == len(thresholds))
    for i in range(map_pred.shape[-1]):
        # Per-pixel and per-channel comparison on a threshold to
        # binarize prediction masks:
        map_pred[:, :, i] = (map_pred[:, :, i] > thresholds[i]).astype('uint8')
        
    if closing:
        map_pred = refine_tree_mask(map_pred)
    return map_pred



def refine_tree_mask(tree_mask, meta_data = None):
    """
    Apply a mortphological operation of closing
    """
    # Ensure the tree_mask is binary first
    assert len(np.unique(tree_mask.flatten())) == 2
    
    if meta_data is not None:
        tree_mask = np.expand_dims( morphology.binary_closing(tree_mask[:,:,0], 
                                      selem = morphology.disk(radius = int(2/meta_data['transform'][0]))), 
            axis=-1)
    else:
        tree_mask = np.expand_dims( morphology.binary_closing(tree_mask[:,:,0], 
                                      selem = morphology.disk(radius = 1)), 
            axis=-1)
    return (tree_mask * 255).astype('uint8')
    



def compute_performance(GT, pred):
    
    
    GT, pred = geo_utils.align_maps(GT, pred)
    
    ground_truth_flat = GT.ravel()/255
    prediction_flat = pred.ravel()

    # --Calculate evaluation metrics
    accuracy = accuracy_score(ground_truth_flat, prediction_flat)
    print("\nAccuracy: {:.2f}".format(accuracy))
    
    precision = precision_score(ground_truth_flat, prediction_flat)
    print("Precision: {:.2f}".format(precision))
    
    recall = recall_score(ground_truth_flat, prediction_flat)
    print("Recall: {:.2f}".format(recall))
    
    f1 = f1_score(ground_truth_flat, prediction_flat)
    print("F1-Score: {:.2f}".format(f1))
    
    roc_auc = roc_auc_score(ground_truth_flat, prediction_flat)
    print("ROC AUC: {:.2f}".format(roc_auc))

    # Jaccard Index (IoU)
    jaccard_index = jaccard_score(ground_truth_flat, prediction_flat)
    print("Jaccard Index (IoU): {:.2f}".format(jaccard_index))

    
    
    
    
    
    



















