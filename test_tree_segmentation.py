import VIRASS as ges
import matplotlib.pyplot as plt
import numpy as np



# Load the ground truth
GT_tree_mask, meta_data_GT = ges.io.open_geoTiFF("_data/trees/tree_segmentation/GT/GTtree.tif")

# Morhological closing
GT = ges.tree_segmentation.refine_tree_mask(GT_tree_mask, meta_data_GT)
ges.io.export_GEOtiff("_data/trees/tree_segmentation/GT/GTtree_closed.tif", GT, meta_data_GT)


if not "SAT_image" in locals():
    SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area.tif") 


tree_segmenter = ges.tree_segmentation.TreeSegmenter(config_file = "tree_segmenter_config.yaml")
tree_segmenter.print_attributes()


tree_segmenter.load_X_map("_data/WorldView_area.tif")
tree_segmenter.load_y_map("_data/trees/tree_segmentation/GT/GTtree_closed.tif")
tree_segmenter.build_model()
tree_segmenter.plot_training_history()

tree_map_pred = tree_segmenter.generate_tree_map(SAT_image)

tree_pred_bn = ges.tree_segmentation.binarize_treeMap(tree_map_pred, thresholds = [50])
ges.io.export_GEOtiff("_data/trees/tree_segmentation/tree_pred_bn.tif", tree_pred_bn, meta_data)



