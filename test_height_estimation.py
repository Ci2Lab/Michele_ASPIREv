import VIRASS as ges
import matplotlib.pyplot as plt
import numpy as np


if not "SAT_image" in locals():
    SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area.tif") 


nDSM_modeler = ges.nDSM_estimation.nDSMmodeler(config_file = "nDSMmodeler_config.yaml")
nDSM_modeler.print_attributes()


nDSM_modeler.load_X_map("_data/WorldView_area.tif")
nDSM_modeler.load_y_map("_data/trees/3d_modeling/GT/warping/nDSM_warped.tif")
nDSM_modeler.build_model()


nDSM_modeler.plot_training_history()

nDSM_map = nDSM_modeler.generate_nDSM(SAT_image)
ges.io.export_GEOtiff("_data/trees/3d_modeling/nDSM_pred.tif", nDSM_map, meta_data)



