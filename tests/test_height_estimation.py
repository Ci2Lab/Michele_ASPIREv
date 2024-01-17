import VIRASS as ges
import matplotlib.pyplot as plt
import numpy as np


# Preprocessing: clip the nDSM to cover the satellite image
# GT_nDSM, meta_data = ges.io.open_geoTiFF("_data/trees/3D_modeling/GT/warping/nDSM_warped.tif")
# alpha = np.clip(ges.io.open_geoTiFF("_data/alpha_channel.tif", with_meta_data = False), 0,1)
# GT_nDSM, alpha = ges.geo_utils.align_maps(GT_nDSM, alpha)
# GT_nDSM = GT_nDSM * alpha
# ges.io.export_GEOtiff('_data/trees/3D_modeling/GT/warping/nDSM_warped_alpha.tif', GT_nDSM, meta_data)



nDSM_modeler = ges.nDSM_estimation.nDSMmodeler(config_file = "nDSMmodeler_config.yaml")
nDSM_modeler.print_attributes()


nDSM_modeler.load_X_map("_data/WorldView_area_training.tif")
nDSM_modeler.load_y_map("_data/trees/3D_modeling/GT/GT_nDSM_training.tif")
nDSM_modeler.build_model()
nDSM_modeler.plot_training_history()


# Prediction
SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area_testing.tif")
nDSM_map = nDSM_modeler.generate_nDSM(SAT_image)
# ges.io.export_GEOtiff("_data/trees/3d_modeling/nDSM_pred_testing.tif", nDSM_map, meta_data)


# # Evaluation
GT_test = ges.io.open_geoTiFF("_data/trees/3D_modeling/GT/GT_nDSM_testing.tif", with_meta_data = False)
nDSM_pred = ges.io.open_geoTiFF("_data/trees/3d_modeling/nDSM_pred_testing.tif", with_meta_data = False) 
ges.nDSM_estimation.compute_performance(GT_test, nDSM_pred)



