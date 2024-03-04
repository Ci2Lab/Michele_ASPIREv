"""
@author: Michele Gazzea
"""

# import ogr, osr
import numpy as np
from PIL import Image

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from matplotlib import cm
import time
import yaml 
import shutil
import os
import warnings
import functools

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
           
def read_config(config_path):
    """Read config yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader = yaml.FullLoader)
    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))
    return config
            

def dict_to_yaml(config : dict):
    file = open("config.yaml","w+")
    yaml.dump(config, file, allow_unicode=True)
    file.close()
    
    
def is_channel_first(SAT_image : np.array):
    """Test whether a satellite image has the channels as first dimensions
    """
    return np.argmin(SAT_image.shape) == 0 

def is_channel_last(SAT_image : np.array):
    """Test whether a satellite image has the channels as last dimensions
    """
    return np.argmin(SAT_image.shape) == len(SAT_image.shape)-1 


def convert_to_channel_last(SAT_image : np.array, verbose = True):
    """ Convert a satellite image from channel-first to channel-last
    """
    assert len(SAT_image.shape) == 3
    
    if is_channel_first(SAT_image):
        old_shape = SAT_image.shape
        SAT_image = np.transpose(SAT_image, (1,2,0))
        new_shape = SAT_image.shape
        if verbose:
            print("Converting to channel_last: {} --> {}".format(old_shape, new_shape))
    else:
        if verbose:
            print("image is already channel_last" + str(SAT_image.shape))
    return SAT_image
        
 
    
def convert_to_channel_first(SAT_image : np.array, verbose = True):
    """ Convert a satellite image from channel-last to channel-first 
    """
    assert len(SAT_image.shape) == 3
    if is_channel_last(SAT_image):
        old_shape = SAT_image.shape
        SAT_image = np.transpose(SAT_image, (2,0,1))
        new_shape = SAT_image.shape
        if verbose:
            print("Converting to channel_first: {} --> {}".format(old_shape, new_shape))       
    else:
        if verbose:
            print("image is already channel_first" + str(SAT_image.shape))
    return SAT_image
            
   
        
def normalizeArray(array, new_range):     
    m = (new_range[1]-new_range[0])/(array.max() - array.min())
    array = m*(array - array.max()) + new_range[1]
    return array


def rescaleArray(array, old_range, new_range):     
    m = (new_range[1]-new_range[0])/(old_range[1] -old_range[0])
    array = m*(array - old_range[1]) + new_range[1]
    return array



def combine_arrays(array1, array2, mode="average"):
    """"
    Average two arrays, array 2 
    """
    assert array1.shape == array2.shape
    
    if mode == "average":
        array = (0.5*array1 + 0.5*array2).astype(array1.dtype)
    
    elif mode == "add":
        #TODO
        pass
    return array    
    





class ImageGui(tk.Tk):
    """
    GUI for visualizing the dataset as numpy array.
    """
    
    def __init__(self, master, X, Y, Z=None):
        """
        Initialise GUI
        """   
        
        # So we can quit the window from within the functions
        self.master = master
        
        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()
        
        # Start at the first file name
        self.index = 0
        self.n_images = X.shape[0]
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        
        # Set empty image container
#        self.image_raw = None
        self.image = None
        self.image2 = None
        self.image3 = None
        self.image_panel = tk.Label(frame)
        self.image_panel2 = tk.Label(frame)
        self.image_panel3 = tk.Label(frame)
        
        # Set image container to first image
        self.set_image()
            
        self.buttons = [] 
                            
        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im 50", width=10, height=2, fg="purple", command=lambda l=0: self.show_prev_image50()))
        self.buttons.append(tk.Button(frame, text="prev im", width=10, height=2, fg="purple", command=lambda l=0: self.show_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", width=10, height=2, fg='purple', command=lambda l=0: self.show_next_image()))
        self.buttons.append(tk.Button(frame, text="next im 50", width=10, height=2, fg='purple', command=lambda l=0: self.show_next_image50()))
        ###
        
        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)  
        self.progress_label.grid(row=2, column=1, sticky='we')
                                                                   
        # Place buttons in grid
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')
            frame.grid_columnconfigure(ll, weight=1)        
        
        
         # Place the image in grid
        self.image_panel.grid(row=1, column=0, sticky='we')
        self.image_panel2.grid(row=1, column=1, sticky='we')
        self.image_panel3.grid(row=1, column=2, sticky='we')
        
                
#******************************************************************************    

    def set_image(self):
        """
        Helper function which sets a new image in the image view
        """
        #X  (RGB image)
        image = self.X[self.index,:,:,0:3].astype('uint8')
        imagePIL = Image.fromarray(image).resize((200,200), Image.NEAREST)
        self.image = ImageTk.PhotoImage(imagePIL, master = self.master)
        self.image_panel.configure(image=self.image) 
        
        
        #Y
        if len(self.Y.shape) == 3:
            image2 = self.Y[self.index,:,:].astype('uint8')
            imagePIL2 = Image.fromarray( cm.viridis(image2, bytes=True)).resize((200,200), Image.NEAREST)

        else:
            image2 = self.Y[self.index,:,:,0:3].astype('uint8')
            # imagePIL2 = Image.fromarray(image2).resize((200,200), Image.NEAREST)
            imagePIL2 = ImageOps.grayscale(Image.fromarray(image2).resize((200,200), Image.NEAREST) )
        self.image2 = ImageTk.PhotoImage(imagePIL2, master = self.master)
        self.image_panel2.configure(image=self.image2) 
        
        
        if isinstance(self.Z, ( np.ndarray)):
            if len(self.Z.shape) == 3:
                image3 = self.Z[self.index,:,:].astype('uint8')
                # imagePIL3 = Image.fromarray( cm.viridis(image3, bytes=True)).resize((200,200), Image.NEAREST)
                imagePIL3 = ImageOps.grayscale(Image.fromarray(image3).resize((200,200), Image.NEAREST) )
            else:
                image3 = self.Z[self.index,:,:,0:3].astype('uint8')
                imagePIL3 = Image.fromarray(image3).resize((200,200), Image.NEAREST)
            self.image3 = ImageTk.PhotoImage(imagePIL3, master = self.master)
            self.image_panel3.configure(image=self.image3)
        
         
        
    def show_prev_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string)
        
        if self.index >= 0:
            self.set_image()
        else:
        #   self.master.quit()
            self.master.destroy()
            
    def show_prev_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 50
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string)
        
        if self.index >= 0:
            self.set_image()
        else:
        #   self.master.quit()
            self.master.destroy()

            
    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string) 

        if self.index < self.n_images:
            self.set_image()
        else:
#            self.master.quit()
            self.master.destroy() 
            
    def show_next_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 50
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string) 
        
        if self.index < self.n_images:
            self.set_image()
        else:
        #            self.master.quit()
            self.master.destroy()
    


def confirmCommand(text):
    text = text + " (y/n)"
    check = input(text)
    if check.lower() in ["y", "yes"]:
        return True
    else:
        return False
    
 
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        print("Time for " + str(func.__name__) + " is " + str(time.time() - start))
        return output
    return wrapper



def open_file(title=""):
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()   # this supress the tk window
    
    filename = filedialog.askopenfilename(parent=window,
                                      initialdir="",
                                      title=title,
                                      filetypes = (("tif images", ".tif"),
                                                   ("numpy arrays", ".npy"),
                                                   ("shapefiles", ".shp"),
                                                   ("geopackage", ".gpkg"),
                                                   ("all", "*")))
    # Here, window.wm_attributes('-topmost', 1) and "parent=window" argument 
    # help open the dialog box on top of other windows
    return filename


def open_dir(title=""):
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()   # this supress the tk window
    
    dirname = filedialog.askdirectory(parent=window,
                                      initialdir="",
                                      title=title)
    # Here, window.wm_attributes('-topmost', 1) and "parent=window" argument 
    # help open the dialog box on top of other windows
    return dirname + "/"
    

def export_array_as_image(array : np.array, filename = "output.png"):
    assert array.dtype == "uint8"
    im = Image.fromarray(array)
    im.save(filename)
    
    
def delete_tmp_folder(tmp_folder):
    """ Delete a temporarly folder. 
    Examples of tmp folder are the ones created to split too large images 
    for the tree segmentation or crown extraction
    """
    # delete the tmp folder
    print("--deleting tmp_folder: {}".format(tmp_folder))
    if os.path.isfile(tmp_folder) or os.path.islink(tmp_folder):
        os.unlink(tmp_folder)
    elif os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
    
    
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    