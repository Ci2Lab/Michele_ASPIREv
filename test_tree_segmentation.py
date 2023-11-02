import VIRASS as ges
import matplotlib.pyplot as plt
import numpy as np



tree_segmenter = ges.tree_segmentation.TreeSegmenter(config_file = "tree_segmenter_config.yaml")
tree_segmenter.print_attributes()


tree_segmenter.load_X_map("_data/WorldView_area_training.tif")
tree_segmenter.load_y_map("_data/trees/tree_segmentation/GT/GT_tree_training.tif")

tree_segmenter.build_model()
tree_segmenter.plot_training_history()


# Prediction
SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area_testing.tif") 
tree_map_pred = tree_segmenter.generate_tree_map(SAT_image)
ges.io.export_GEOtiff("_data/trees/tree_segmentation/tree_pred_testing_" + tree_segmenter.config['config_training']['model_name'] + ".tif", tree_map_pred, meta_data)
tree_pred_bn = ges.tree_segmentation.binarize_treeMap(tree_map_pred, thresholds = [100])


# Evaluation
GT_test = ges.io.open_geoTiFF("_data/trees/tree_segmentation/GT/GT_tree_testing.tif", with_meta_data = False)
tree_pred = ges.io.open_geoTiFF("_data/trees/tree_segmentation/tree_pred_testing.tif", with_meta_data = False) 
tree_pred_bn = ges.tree_segmentation.binarize_treeMap(tree_pred, thresholds = [100])
ges.tree_segmentation.compute_performance(GT_test, tree_pred_bn)





