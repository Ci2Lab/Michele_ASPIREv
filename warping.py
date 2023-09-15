import rasterio
import GridEyeS_lib as ges
import PIL
import numpy as np
import os

    

class Warper():
    def __init__(self,):
        
        """ Attributes """
        self.meta_data = dict()
        self.nDSM = None
        self.warping_completed = False
    
    
    def get_meta_data(self, file_path):
        if len(self.meta_data) == 0:
            if not os.path.isfile(file_path):
                relocate = ges.utils.confirmCommand("File: <"+ file_path + "> does not exist. Do you want to locate it in other directories?")
                if relocate:
                    # ask for new filename
                    file_path = ges.utils.open_file(title = "Open GeoTIFF image")
                else: 
                    raise FileNotFoundError()
                
            #--Open GeoTIFF file and extract meta data
            file_src = rasterio.open(file_path)
            meta_data = file_src.profile    
            self.meta_data = meta_data
            file_src.close()                 
        else:
            print("meta data already defined")
     
            
    def get_nDSM(self, file_path):        
        if not os.path.isfile(file_path):
            relocate = ges.utils.confirmCommand("File: <"+ file_path + "> does not exist. Do you want to locate it in other directories?")
            if relocate:
                # ask for new filename
                file_path = ges.utils.open_file(title = "Open GeoTIFF image")
            else: 
                raise FileNotFoundError()

        #--Open GeoTIFF file and extract the image
        file_src = rasterio.open(file_path)
        meta_data = file_src.profile    
        bands = range(1, meta_data['count']+1)
        nDSM = file_src.read(bands)
        file_src.close()
        self.nDSM = nDSM
    
    
    def prepare_nDSM(self, height_to_pixel_factor = 0.15):
        """ Process the DHM. 
        1-Clip the DHM from 0 to 38.25  (0.25*255)
        2-convert height values into pixel intensity values
        3-save it as .png, so it can be opened with Photoshop for the warping procedure 
        """
        nDSM = np.clip(self.nDSM, 0, height_to_pixel_factor*255)[0,:,:]
        nDSM = (np.round(nDSM / height_to_pixel_factor)).astype('uint8')
        im = PIL.Image.fromarray(nDSM)
        if ges.utils.confirmCommand("Save it as PNG?"):
            im.save("warping/DHM_image.png")

        
        
    def load_PNG_warped(self, file_name):
        if self.warping_completed:
            if not os.path.isfile(file_name):
                file_name = ges.utils.open_file("Locate warped PNG")
            warped_img = np.array(PIL.Image.open(file_name))
            self.nDSM = np.transpose(warped_img, (2,0,1))[0,:,:]
        else:
            print("Warping process not completed")
        
    
    def pixel_to_height(self, height_to_pixel_factor = 0.15):
        self.nDSM = (self.nDSM * height_to_pixel_factor).astype('float32')
        
        
    def export_warped_DHM(self,):
        if self.warping_completed:
            self.meta_data['nodata'] = None
            # Convert pixel value into height again
            ges.io.export_GEOtiff("nDSM_warped.tif", self.nDSM, self.meta_data)
        else:
            print("Warping process not completed")
        
        
            

if __name__ == "__main__": 
    warper = Warper()      
    # Get meta data
    warper.get_meta_data("_data/trees/3D_modeling/GT/warping/WV_image.tif")
    
    # Get DHM
    # warper.get_nDSM("warping/DHM.tif")
    
    # # Prepare the DHM
    # warper.prepare_nDSM()
    
    # # Warping with Photoshop
    warper.warping_completed = True
    
    # # Load 
    warper.load_PNG_warped("warping/nDSM_warped.png")
    warper.pixel_to_height()
    warper.export_warped_DHM()
    

