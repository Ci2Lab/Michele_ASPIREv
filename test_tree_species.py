import VIRASS as ges
import matplotlib.pyplot as plt
import numpy as np



if not "R_ref" in locals():     
    # Load Satellite image 
    SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area.tif")  
    
    # Load the NIBIO dataset
    R_ref = ges.io.open_geoTiFF("_data/trees/tree_species/NIBIO.tif", with_meta_data = False)
    
    # Load binary tree_mask
    tree_mask_bn  = ges.io.open_geoTiFF("_data/trees/tree_segmentation/tree_mask_bn_1_2.tif", with_meta_data = False)


tree_species_classifier = ges.tree_species_classification.TreeSpeciesClassifier(config_file = "tree_species_classifier_config.yaml")
tree_species_classifier.print_attributes()


R_relabeled = tree_species_classifier.relabeling(R_ref, tree_mask_bn, SAT_image, meta_data)
ges.io.export_GEOtiff(tree_species_classifier.working_dir + "R_relabeled_2.tif", R_relabeled, meta_data)

tree_species_classifier.load_X_map("_data/WorldView_area.tif")
tree_species_classifier.load_y_map("_data/trees/tree_species/R_relabeled_2.tif")
tree_species_classifier.build_model()


tree_species_map_pred = tree_species_classifier.generate_tree_species_map(SAT_image)

tree_species_map_quantized = ges.tree_species_classification.quantize_tree_species_map(tree_species_map_pred)
ges.io.export_GEOtiff("_data/trees/tree_species/R_refined_2_Q.tif", tree_species_map_pred, meta_data)
