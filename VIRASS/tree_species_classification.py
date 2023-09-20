import numpy as np
import os
import geopandas
import pandas
import pickle
from shapely.geometry import Point
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from . import RSClassifier
from . import utils
from . import architectures
from . import geo_utils
from . import io
from .import stitching

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import LearningRateScheduler


def pre_process_R_ref(R_ref, source = "NIBIO"):
    """
    Pre-process the tree reference dataset. 
    Several required steps might be implemented depending on the dataset.
    For example for NIBIO, label 3 (mixture) and 4 (coniferous) are removed.
    For another use case, other steps should be required
    """
    if source == "NIBIO":
        print("Pre process NIBIO dataset")
        R_ref[np.where(R_ref == 3)] = 0 #we don't consider mixture
        R_ref[np.where(R_ref == 4)] = 0 #we don't consider coniferous
        
    else:
        print("Specify pre-processing steps for the 'tree reference' dataset")    
    return R_ref


def create_baseline(R_ref, T_mask) :
    """
    The baseline R_mask is calculated multiplying the tree reference dataset R_ref (e.g. NIBIO)
    with the tree segmentation map T_mask 
    """
    
    """Quality check:
        T_mask should be (height, width, 1)
        T_ref should be (height, width, 1)"""
    
    assert utils.is_channel_last(R_ref) and utils.is_channel_last(T_mask) 
    print("Create baseline")    
    # Clip T_mask to 0,1
    T_mask = np.clip(T_mask, a_min = 0, a_max = 1) 
    # Multiply
    R_mask = R_ref * T_mask
    return R_mask



@utils.measure_time
def extract_patches_from_baseline(SAT_image, R_mask, meta_data,  
                                  tree_types_legend,
                                  number_patches, radius_patch = 6, 
                                  saving_folder = "",
                                  save_patch = True,
                                  save_locations = True):

    
    # if the folder does not exist, create it
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
        
    dataset_present = False 
    overwrite = False
    # Are there exisintg files?
    if not len(os.listdir(saving_folder)) == 0:
        dataset_present = True
        if utils.confirmCommand("Patches are already present. Overwrite them?"):
            overwrite = True
    
         
    if not dataset_present or overwrite:
        
        # column_names = ["geometry", "tree_species"]
        # geographical_locations = geopandas.GeoDataFrame(columns = column_names)
        geographical_locations = []
        
        for tree_type_name in tree_types_legend.keys():
            print("--Extracting patches for <" + tree_type_name + ">")
            tree_patches, loc = _extract_tree_patches_from_baseline(SAT_image, R_mask, meta_data, 
                                                        tree_type_ID = tree_types_legend[tree_type_name], 
                                                        number_patches = number_patches[tree_type_name],
                                                        radius_patch = radius_patch)   
            if save_patch:
                # Save the patches
                np.save(saving_folder + tree_type_name, tree_patches)
            
            if save_locations:
                # Save the locations into a geopandas DataFrame
                geographical_locations.append(geopandas.GeoDataFrame( {'geometry' : map(Point, zip(loc[:,0], loc[:,1])),
                                                                       'tree_species' : tree_type_name}))         
        
        if save_locations:
            geographical_locations = pandas.concat(geographical_locations)
            geographical_locations.crs = meta_data['crs']
            geographical_locations.to_file(saving_folder + "locations_patches.gpkg", driver = 'GPKG')
            




def _extract_tree_patches_from_baseline(SAT_image, R_mask, meta_data,  
                                 tree_type_ID,
                                 number_patches,
                                 radius_patch = 6):
    """
    Create an array of patches (from SAT_image) selected from the NIBIO: <tree_type_name>
    tree_type_ID: number correspondent to the species, e.g. spruces=1 
    """
    
               
    def _task_position(ind, meta_data):
        """Given an array of indeces of all the possible position centers, pick randomly one"""
        # Select a random center to extract a patch from
        point = np.random.randint(low = 0, high = len(ind[0]) ) 
        row = ind[0][point]
        col = ind[1][point]
        # Check if the center is too close to the SAT border
        while not (row > radius_patch and row < meta_data['height'] - radius_patch \
                and col > radius_patch and col < meta_data['width'] - radius_patch): 
            point = np.random.randint(low = 0, high = len(ind[0]) ) 
            row = ind[0][point]
            col = ind[1][point]
        return row, col
    
    # Create an array to store the patches    
    tree_patches = np.empty( (number_patches, 2*radius_patch, 2*radius_patch, SAT_image.shape[-1]) ).astype('uint8')        
    # Create an array to store the locations of the patches
    loc = np.empty( (number_patches, 2) )
       
    #Search for all the <Tree> indeces
    ind = np.where(R_mask == tree_type_ID)
    
    if ind[0].size == 0:
        print("no patches with tree_ID: " + str(tree_type_ID))
        tree_patches = None
        loc = None
    else:
        iteration = 0
        while iteration < number_patches:
            row, col = _task_position(ind, meta_data) 
            tree_patches[iteration, :,:,:] = SAT_image[row-radius_patch:row+radius_patch, col-radius_patch:col+radius_patch, :]
                          
            #Compute geographical location: not required, useful to visualize the centers in QGIS in case
            loc[iteration,:] = meta_data['transform'][0] * col + meta_data['transform'][2], \
                    meta_data['transform'][4] * row + meta_data['transform'][5]
               
            iteration = iteration + 1
            
    return tree_patches, loc




def load_tree_type_patches(tree_types, saving_folder):
    """
    Load tree patches as a dictionary
    """
    # Are there exisintg files?    
    if not len(os.listdir(saving_folder)) == 0:
        tree_patches = {}
        for tree_type in tree_types:
            tree_patches[tree_type] = np.load(saving_folder + tree_type + ".npy")
        
        return tree_patches
    else:
        print("Dataset not present")
        return None



def shuffle_tree_patches(tree_patches, tree_types_legend):
    """
    Shuffle patches and corresponding labels, preserving of course the matching
    """
    def _dictionary2array(dictionary, tree_types_legend):
        patches = []
        labels = []
        for tree_type in dictionary.keys(): 
            tmp = dictionary[tree_type]
            patches.append( tmp )
            labels.append( tree_types_legend[tree_type] * np.ones((len(tmp),)) )
         
        patches = np.concatenate(patches, axis=0)  
        labels = np.concatenate(labels, axis=0)
        return (patches, labels)
    
    patches, labels = _dictionary2array(tree_patches, tree_types_legend)
    idx = np.random.permutation(len(patches))
    patches, labels = patches[idx], labels[idx]
    return patches, labels




class FeatureExtractor():
    """ Feature extractor class. Given a patch, it extracts a feature vector for that patch
    Method: 
        - manual: based on manual features (color, texture, etc...)
        - AE: based on normal autoencoders
        - VAE: based on Variational autoencoders
    """      
    def __init__(self, config : dict()):
        print("***\nclass <" + self.__class__.__name__ + "> initialized ***\n")
        
        # Attributes
        self._config = config
        
        # assert "working_dir" in config
    
    
    def compute_features(self, method, patches : np.array):
        assert (len(patches.shape) == 4) and (patches.shape[1] == patches.shape[2])
        print("Tree patches: " + str(patches.shape))
        
        if method.lower() == 'manual':
            print("Extracting manual features")
            features = self.compute_features_MANUAL(patches)
        elif method.lower() == 'ae':
            print("Extracting features via autoencoders")
            features = self.compute_features_AE(patches)
            return features
        elif method.lower() == 'vae':
            print("Extracting features via variational autoencoders")
            features = self.compute_features_VAE(patches)
        return features
    
                
    
    @utils.measure_time   
    def compute_features_MANUAL(self, ): 
        # Method already implemented in TGRS. 
        pass
     
    @utils.measure_time        
    def compute_features_AE(self, patches): 
        input_shape = patches.shape[1::]
        print("Input shape: ", input_shape)        
        
        saving_folder = self._config['working_dir'] + "AE/"
        if not 'autoencoder_name' in self._config:
            autoencoder_name = "autoencoder"
        else:
            autoencoder_name = self._config['autoencoder_name']
        if not 'encoder_name' in self._config:
            encoder_name = "encoder"
        else:
            encoder_name = self._config['encoder_name']
        
            
        # Create the model
        AE = Autoencoder()
        
        if os.path.isfile(saving_folder + autoencoder_name +'.h5'):
            AE.load(saving_folder, autoencoder_name, encoder_name)
        else:
            AE.build_autoencoder(saving_folder, autoencoder_name, encoder_name, patches)
                        
        #compute the features
        features = (AE.compute_features(patches)).reshape(len(patches),-1)
        return features
     
    
    def compute_features_VAE(self,): 
        #TODO
        pass
    
 
    
class Autoencoder():  
    def __init__(self):
        
        print("***class <" + self.__class__.__name__ + "> initialized ***\n")
        
        self.features = None                
        self.autoencoder = None
        self.encoder = None
        
        
    def load(self, saving_folder, autoencoder_name, encoder_name):
        if os.path.isfile(saving_folder + autoencoder_name +'.h5'):
            print("-- Loading existing autoencoder")    
            autoencoder = keras.models.load_model(saving_folder + autoencoder_name +'.h5')
            encoder = keras.models.load_model(saving_folder + encoder_name +'.h5')                
            self.autoencoder = autoencoder
            self.encoder = encoder
            print("--autoencoder and encoder loaded")
        else:
            print("{} does not exist in {}".format(autoencoder_name, saving_folder))
        
        
    def build_autoencoder(self, saving_folder, autoencoder_name, encoder_name, training_patches):
        input_shape = training_patches.shape[1::]
        print("Input shape: ", input_shape) 
            
        # Instantiate autoencoder    
        autoencoder, encoder = architectures.autoencoder_model(input_shape = input_shape)        
        self.autoencoder = autoencoder
        self.encoder = encoder
        
        AE_exists = False
        overwrite = False
        #File already exists?
        if os.path.isfile(saving_folder + autoencoder_name +'.h5'):
            AE_exists = True
            if utils.confirmCommand("-An autoencoder with the same name already exists. Do you want to overwrite it?"):
                overwrite = True 
                    
        if (not AE_exists) or overwrite:
            print("Creating and training a new autoencoder")
            history = self._train(training_patches)
            
            # Create folder if not there
            if not os.path.isdir(saving_folder):
                os.makedirs(saving_folder)
            print("Saving autoencoder")    
            autoencoder.save(saving_folder + autoencoder_name +'.h5')
            print("Saving encoder")
            encoder.save(saving_folder + encoder_name +'.h5')
            print("Saving training history")
            with open(saving_folder + autoencoder_name +'_history', 'wb') as file:
                pickle.dump(history.history, file) 
       
        
    def pre_processing(self, X):
        X = X.astype('float32')/255.
        return X
    
    
    def _train(self, X, epochs = 60): #60        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = keras.losses.MeanSquaredError()
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        
        _X = self.pre_processing(X)
        def scheduler(epoch, lr):
            if epoch < 100:
                return lr
            else:
                return lr * np.math.exp(-0.01)
        history = self.autoencoder.fit(_X, _X,  batch_size = 128, epochs=epochs)#, callbacks=[LearningRateScheduler(scheduler)])
        return history
    
    def compute_features(self, patches):
        if self.encoder is not None:
            _patches = self.pre_processing(patches)
            # self.features = self.encoder.predict(_patches)
            return self.encoder.predict(_patches)
        else:
            print("Autoencoder = None")
        
    def recostructImage(self, patches):
        if self.autoencoder is not None:
            _patches = self.pre_processing(patches)
            return (self.autoencoder.predict(_patches)*255).astype('uint8')
        else:
            print("Autoencoder = None")
    



def tSNE_featureVisualization(features_scaled, labels, 
                              config_tse = {'colors_per_class' : {"spruces" : "green", "pines": "blue", "deciduous": "orange"},
                                        'perplexity' : 40,
                                        'tree_types_legend' : {"spruces": 1, "pines": 2, "deciduous": 5},
                                        'method' : "umap"}):
    
    perplexity = config_tse['perplexity']
    colors_per_class = config_tse['colors_per_class']
    tree_types_legend = config_tse['tree_types_legend']
    method =  config_tse['method']
    
        
    def plot_tse( tx, ty, labels, tree_types_legend, colors_per_class ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for tree_type_name in colors_per_class.keys():
            # find the samples of the current class in the data
            tree_type_ID = tree_types_legend[tree_type_name]
            # extract the coordinates of the points of this class only
            current_tx = tx[np.where( labels == tree_type_ID)]
            current_ty = ty[np.where( labels == tree_type_ID)]
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, s = 3, c = colors_per_class[tree_type_name], label = tree_type_name)
        ax.legend(loc='best');         plt.legend(fontsize=15)
        plt.tight_layout()
       
    if config_tse['method'] == "tsne":
        from sklearn.manifold import TSNE
        #Compute t-SNE of high-dimensional features
        embedding = TSNE(n_components = 2, perplexity = perplexity, learning_rate='auto', init='random').fit_transform(features_scaled)
    
    elif config_tse['method'] == "umap":
        import umap
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(features_scaled)
        
    tx = utils.normalizeArray(embedding[:, 0], (0,1))
    ty = utils.normalizeArray(embedding[:, 1], (0,1))
    
    plot_tse( tx, ty, labels, tree_types_legend, colors_per_class)


    
    
def clustering(features, n_clusters = 30):
    """
    Group the features close to each other in the feature space in the same cluster and 
    assign each point in the feature space a cluster label (from 0 to n_clusters)

    Parameters
    ----------
    features : np.array  (Npoints, Nfeatures)
        

    Returns
    -------
    clusters_labels : np.array
         
    """
    # Spectral clustering can be an option to look into 
    # clustering = SpectralClustering(n_clusters=2,assign_labels='discretize', random_state=0).fit(features_scaled)
    kmeans = KMeans(n_clusters, random_state=0).fit(features)
    clusters_labels = kmeans.predict(features)
    return kmeans, clusters_labels


    
    

def semantic_costraint(clusters_labels, classes_labels, n_clusters, n_classes = 3):
    """
    Learn and assign to each cluster (0,1,...,n_clusters-1) a class (1,...,n_classes) using purity measurements. 
    The idea is thah points close to each other in the feature space should have the same class label.
    This assumption is used to relabel some features with 'wrong' label in teh relabeling process 
    (implemented in another function).
    The 'purity' P(cluster, class) is defined as the ratio between the number of points having class <class> and 
    the total number of points in the cluster <cluster>.   
    Parameters
    ----------
    clusters_labels : TYPE
        DESCRIPTION.
    classes_labels : TYPE
        DESCRIPTION.
    n_clusters : TYPE
        DESCRIPTION.
    n_classes : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    cluster_class_mapping : np.array (n_clusters,)
        A vector that specify a class for each cluster 

    """
    
    S = np.empty(n_clusters)
    PURITY = np.zeros((n_clusters, n_classes))

    tree_label = np.unique(classes_labels)
    
    # Compute the purity per each cluster
    for cluster in range(0, n_clusters):
        pt = np.where(clusters_labels==cluster)[0]
        #for each cluster: count how many points we have in such cluster
        S[cluster] = len(pt)
        
        for _class in range(0, n_classes): 
            class_label = tree_label[_class]
            #per each class_label, count how many points per cluster
            
            PURITY[cluster, _class] = len(np.where( classes_labels[pt] == class_label)[0]) / S[cluster]


    #assign to all points in cluster the class with highest purity
    cluster_class_mapping = np.empty(n_clusters)
    for cluster in range(0, n_clusters): 
        pt = np.where(clusters_labels==cluster)[0]
        cluster_class_mapping[cluster] = tree_label[np.argmax(PURITY[cluster,:])]
        # classes_labels[pt] = cluster_class_mapping[cluster] <-- not needed
   
    # return classes_labels, cluster_class_mapping
    return cluster_class_mapping   



def spatial_costraint(array, patch_size, group_size):
    assert group_size%patch_size == 0 
    
    for row in range(0, array.shape[0], group_size):
        for col in range(0, array.shape[1], group_size):
            group = array[row:row+group_size, col:col+group_size]
            if np.sum(group) == 0:
                pass
            else:
                values, counts = np.unique(group, return_counts=True)
                
                if values[0] == 0: 
                    #if the most common value is zero, consider the second most common value
                    value = values[np.argmax(counts[1:])+1]
                else:
                    value = values[np.argmax(counts)]
                
                group[np.where(group != 0)] = value
                array[row:row+group_size, col:col+group_size] = group
    return array





class TestClass(RSClassifier.RSClassifier):
    def __init__(self, config_file: str = 'config.yml'):
        super().__init__()
        
        # Check if the file exists
        if not os.path.exists(config_file):
            config_file = utils.open_file("Open configuration file (yaml format)")
        
        # Check if the file has a YAML extension
        if os.path.splitext(config_file)[1].lower() == '.yaml':
            self.config = utils.read_config(config_file)
        else:
            raise ValueError("Wrong format for the configuration file (must be YAML)")


        # Define a dictionary of default values
        default_values = {
            'config_AE': {
                'autoencoder_name': 'Default_AE',
                'encoder_name': 'Default_E',
                'method': 'Default_method',
            },
            'config_relabeling': {
                'plot_uMAP': False,
                'number_patches': {
                    'deciduous': 2000,
                    'pines': 2000,
                    'spruces': 2000
                },
                'radius_patch': 6,
                'tree_types_legend': {
                    'deciduous': 5,
                    'pines': 2,
                    'spruces': 1
                },
                'n_cluster': 30,
            },
            'config_training': {
                'architecture': 'Default_architecture',
                'patch_number': 20000,
                'patch_radius': 40,
                'model_name': 'Default_model',
                'n_epochs': 200,
                'lr': 0.001,
            }
        }

        # Update the loaded configuration with missing keys and default values
        self.config = self.update_with_defaults(self.config, default_values)

    def update_with_defaults(self, config, defaults):
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = self.update_with_defaults(config[key], value)
        return config

               
   
            

class TreeSpeciesClassifier(RSClassifier.RSClassifier):
        
    # def __init__(self, *arg, **config):
    #     super().__init__()
    #     # https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    #     for dictionary in arg:
    #         for key in dictionary:
    #             setattr(self, key, dictionary[key])
    #     for key in config:
    #         setattr(self, key, config[key])
    
    def __init__(self, config_file: str = 'config.yml'):
        super().__init__()
        
        # Check if the file exists
        if not os.path.exists(config_file):
            config_file = utils.open_file("Open configuration file (yaml format)")
        
        # Check if the file has a YAML extension    
        if os.path.splitext(config_file)[1].lower() == '.yaml':
            self.config = utils.read_config(config_file)
        else:
            raise ValueError("Wrong format for the configuration file")
            
        
    # TODO: Set default values in case some important parameters are not passed in by the config.yaml
                
    
    
    def _fix_label(self, R_map):
        """ NIBIO classes by deafult have the following legend:
            spruces: 1
            pines: 2
            deciduous: 5
            
        However, when train models using categorical cross-entropy, having a class 5 and no classes 3 or 4 might lead to
        issues or unnecessary moemory usage.
        Here, we convert all classes in sequential: 0,1,2,3
        """
        R_map[np.where(R_map == 5)] = 3
        return R_map
    
    
    def relabeling(self, R_ref, tree_mask_bn, SAT_image, meta_data):
        
        config_relabeling = self.config['config_relabeling']
        # Pre-process R_ref
        R_ref = pre_process_R_ref(R_ref, source = "NIBIO")
        
        # Create baseline R_mask
        R_mask = create_baseline(R_ref, tree_mask_bn)
        
        # Extract random patches from R_mask
        patch_folder = self.working_dir + "patches/" 

        extract_patches_from_baseline(SAT_image, R_mask, meta_data,  
                                          config_relabeling['tree_types_legend'],
                                          config_relabeling['number_patches'],
                                          saving_folder = patch_folder)
    
        tree_patches = load_tree_type_patches(config_relabeling['tree_types_legend'], patch_folder)
        # Shuffle them
        tree_patches, labels = shuffle_tree_patches(tree_patches, config_relabeling['tree_types_legend'])
    
        # Compute features from patches
        config_feature_extractor = self.config['config_AE'] 
        config_feature_extractor['working_dir'] = self.working_dir
        feature_extractor = FeatureExtractor(config_feature_extractor)
        features = feature_extractor.compute_features(method = "ae", patches = tree_patches)
        
        # Scale features 
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        if self.config['config_relabeling']['plot_uMAP']:
            # Plot uMAP of the features
            tSNE_featureVisualization(features_scaled, labels)

        # Relabel features based on their position in the feature space
        n_clusters = self.config['config_relabeling']['n_cluster']
        kemans, clusters_labels = clustering(features_scaled, n_clusters)

        # Semantic costraint:
        print("--Semantic costraint")
        cluster_class_mapping = semantic_costraint(clusters_labels, labels, n_clusters, n_classes=3)

        # Relabeling:
        print("--Relabeling over the entire area")
        # Now that we learned the mapping between clusters <-> classes, 
        # we can extract patches -> features over the entire area and relabel them.
        tilesGen = geo_utils.tilesGenerator(image = SAT_image, stepSize = 2 * config_relabeling['radius_patch'])      
        tilesGen.image2tiles()
        tiles = tilesGen.tiles

        features_area = feature_extractor.compute_features(method = "ae", patches = tiles)

        # scale them
        features_area_scaled = scaler.transform(features_area)
           
        # predict the clusters using the fitted Kmeans
        labels_area = kemans.predict(features_area_scaled)
                
        # assign each point of the cluster j the class c found in the "cluster_class_mapping".
        for cluster in range(0, n_clusters):
            labels_area[np.where(labels_area == cluster)] = cluster_class_mapping[cluster]

        R_relabeled = ( tilesGen.tiles2image(labels_area) ).astype('uint8')

           
        # Load the tree map and filter the R_relabeled
        R_relabeled = R_relabeled * tree_mask_bn[:,:,0]

        R_relabeled = spatial_costraint(R_relabeled, patch_size = config_relabeling['radius_patch'], group_size = 3*config_relabeling['radius_patch'])
        
        # Check and fix labels for the map. Labels should be 0,1,2,3,...
        R_relabeled = self._fix_label(R_relabeled)
        return R_relabeled
        
   
    
    
    def instantiate_model(self, input_shape, architecture):
        model_path = self.working_dir + self.config['config_training']['model_name'] +'.h5'
        
        if os.path.isfile(model_path): 
            print("--Model already exist. Call <load_model> class method")
        else:  
            
            if architecture == "unet":                                                                
                model = architectures.unet(input_shape = input_shape, n_classes = 4)
            
            else:
                raise NotImplementedError()
           
            opt = tf.keras.optimizers.Adam(learning_rate = self.config['config_training']['lr']) 
            loss = tf.keras.losses.SparseCategoricalCrossentropy()#ignore_class = 0)
            model.compile(loss = loss, 
                        optimizer = opt, 
                        metrics=['accuracy'])            
                        # metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
                        # metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), 
                        #            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
            print("Model initialized\n")
            # model.summary()
            self.DLmodel = model
        

    def _process_dataset(self, X, y):
        X = X / 255.0
        
        # y is reshaped because the model is trained using sparse categorical cross-entropy
        y = y[:,:,:,0]        
        y = y.reshape((len(X),-1))
        return X, y
    
    
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
            
            
            # TODO: Re-implement the pipeline using the tensorflow.keras.utils.Sequence to deal with GPU memory issues 
            # https://stackoverflow.com/questions/62916904/failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
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
            X, y = self._process_dataset(X, y)
            
            print("--Training model on the following dataset: \n")
            print(X.shape, y.shape)
            self.train_model(X, y, model_path)
        
        
        
    def generate_tree_species_map(self, image):
        if self.DLmodel is not None:
            # How much memory would the tiles consume roughly? in GB
            nb_classes = 4
            window_size = 2 * self.config['config_training']['patch_radius']
            subdivisions = 2
            N_tiles_rows = subdivisions * image.shape[0] / window_size
            N_tiles_cols = subdivisions * image.shape[1] / window_size
            # 4 is the factor from uint8 to float32 reppresentation, 1024**3 is the byte->GB conversion
            ESTIMATED_MEMORY = (N_tiles_rows * N_tiles_cols * window_size**2 * image.shape[2] * 4 * nb_classes)/(1024**3)
            print("Estiamted memory {} GB \n". format(ESTIMATED_MEMORY))
            
            # if the data require less than 5 GB, then fit it all
            if ESTIMATED_MEMORY < 5:
                # tree mask can be computed a straight way
                stiched_pred_map = self._generate_tree_species_map(image,  window_size, subdivisions)
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
                        stiched_pred_map.append(self._generate_tree_species_map(part, window_size, subdivisions))
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
             
            
    def _generate_tree_species_map(self, image, window_size, subdivisions):                    
            image = (image / 255).astype('float32')
            print("Segmenting trees with tiles smoothing function")

            stiched_pred_map = stitching.predict_img_with_smooth_windowing(image, window_size, subdivisions, nb_classes=4, 
                                          pred_func = self.DLmodel.predict)
                
            stiched_pred_map = (stiched_pred_map*255).astype('uint8')
            return stiched_pred_map
        
        
    def quantize_tree_species_map(self, tree_species_map):
        """ Convert each pixel to a single value (=class)
        """
        tree_species_map_quantized = np.argmax(tree_species_map, axis = -1).astype('uint8')
        return tree_species_map_quantized

    

    
   
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    