# GreenGuardian

[![Github Actions](https://github.com/weecology/DeepForest/actions/workflows/Conda-app.yml/badge.svg)](https://github.com/weecology/DeepForest/actions/workflows/Conda-app.yml)
[![Documentation Status](https://readthedocs.org/projects/deepforest/badge/?version=latest)](http://deepforest.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/pypi/v/DeepForest.svg)](https://pypi.python.org/pypi/DeepForest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2538143.svg)](https://doi.org/10.5281/zenodo.2538143)


### Conda-forge build status

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-deepforest-green.svg)](https://anaconda.org/conda-forge/deepforest) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/deepforest.svg)](https://anaconda.org/conda-forge/deepforest) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/deepforest.svg)](https://anaconda.org/conda-forge/deepforest) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/deepforest.svg)](https://anaconda.org/conda-forge/deepforest) |

![img](logo.png)

GreenGuardian is a Python package for assessing vegetation-related risk along infrastructure lines. DeepForest implements sub-modules such as
- vegetation segmentation
- tree species classification
- height estimation
- individual tree crown delineation
- tree inventory generation
- risk calculations 

from satellite imagery. 
GreenGuardian comes with prebuilt models. Users can extend this model by annotating and training custom models starting from the prebuilt model.


## Motivation

 Vegetation along the infrastructure lines is one of the major causes of outages, especially along roadways and power lines. Trees can fall after strong winds, disrupting the lines and blocking the roads. If a tree is growing too close to the power line, it might also trigger wildfires.
 Traditional approaches based on visual inspections are extremely time-consuming and very costly.
 
 *GreenGuardian* aims to ease the process of infrastructure monitoring, providing tool to characterize vegetation and calculate thread posing to nearby infrastructure.
  

#### Try Demo using Jupyter Notebook

Incorportating local data will always help prediction accuracy to customize the release model see see [Google colab demo on model training](https://colab.research.google.com/drive/1gKUiocwfCvcvVfiKzAaf6voiUVL2KK_r?usp=sharing)

# Installation

Deepforest can be install using either pip or conda.

## pip

```
pip install greenguardian
```

# Usage

# Use Benchmark release

```Python
from greenguardian import main
m = main.deepforest()
m.use_release()
```

## Train a new model

```Python
m.create_trainer()
m.trainer.fit(m)
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
```
 
## Predict a single image

```Python
#Create a sample 3 band image
image = np.random.random((400,400,3)).astype("float32")
prediction = m.predict_image(image = image)
```

## Predict a large tile

Split the large tile into small pieces, predict each piece and re-assemble

```Python
predicted_boxes = m.predict_tile(raster_path = raster_path,
                                        patch_size = 300,
                                        patch_overlap = 0.5,
                                        return_plot = False)
```

## Evaluate a file of annotations using intersection-over-union as an metric of accuracy

```Python
csv_file = get_data("example.csv")
root_dir = os.path.dirname(csv_file)
results = m.evaluate(csv_file, root_dir, iou_threshold = 0.5)
```

# Config

DeepForest comes with a default config file (deepforest_config.yml) to control the location of training and evaluation data, the number of gpus, batch size and other hyperparameters. This file can be edited directly, or using the config dictionary after loading a deepforest object.

```Python
from deepforest import main
m = main.deepforest()
m.config["batch_size"] = 10
```
Config parameters are documented [here](https://deepforest.readthedocs.io/en/latest/ConfigurationFile.html).

# Tree Detection Benchmark score

Tree detection is a central task in forest ecology and remote sensing. The Weecology Lab at the University of Florida has built a tree detection benchmark for evaluation. After building a model, you can compare it to the benchmark using the evaluate method.

```
git clone https://github.com/weecology/NeonTreeEvaluation.git
cd NeonTreeEvaluation
```
```Python
results = m.evaluate(csv_file = "evaluation/RGB/benchmark_annotations.csv", root_dir = "evaluation/RGB/")
results["box_recall"]
results["box_precision"]
```


 
