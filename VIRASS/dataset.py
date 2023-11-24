import numpy as np
import tensorflow as tf

def _create_dataset(SAT_image, meta_data, patch_number, patch_radius, save_patch = False, export_patch_locations = False):
    """Extract random patches for training from <X_map> (SAT image) and <y_map> (ground truth).
    X_map should be shaped as (heigth, width, Nchannels) 
    y_map should be shaped as (heigth, width, Nchannels = 1) 
    """
            
    radius = patch_radius
    N_patches = patch_number                
    
    # The list implementation is faster than index assign a value within a for loop 
    X = []
    Y = []
     
    height = SAT_image.shape[0]
    width = SAT_image.shape[1]
    #select random row and col
    np.random.seed(0) # make it predictable
    rows = np.random.randint(low = radius, high = height-radius-1, size = (N_patches,))
    cols = np.random.randint(low = radius, high = width-radius-1, size = (N_patches,))
    
    for n in range(0, N_patches):
        row = rows[n]; col = cols[n]
        X.append(SAT_image[row-radius:row+radius, col-radius:col+radius,:])
    
    X = np.asarray(X)
    
    # Create a TensorFlow dataset from the patches
    X = tf.data.Dataset.from_tensor_slices(X)                                  
    return X
        
        
        
        
        
        
        
        
        
        
        
        
        
        