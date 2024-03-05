__author__ = """Michele Gazzea"""
__email__ = 'michele.gazzea@gmail.com'
__version__ = '1.0.0'

print("*** Loading storm-ASPIREv ***")
import h5py # somehow this module needs to be imported when I test locally on my computer.
# On the server, it is automatically loaded from tensorflow/keras. 
# I guess it is something related to versioning issues.  

import warnings as _warnings
from pkgutil import find_loader 

from . import io
from . import utils
from . import geo_utils
from . import RSClassifier
from . import stitching
from . import architectures
from . import tree_segmentation
from . import tree_utils
from . import nDSM_estimation
from . import coregistration
from . import pipeline_utils

from . import lidar # functions to deal with .laz/.las files


# Optional dependencies
if not find_loader('laspy'):
    _warnings.warn('\n*laspy* library is missing. storm-ASPIREv will still work but the `lidar` module will not work.')
                   
if not find_loader('cv2'):
    _warnings.warn('\n*cv2* library is missing. storm-ASPIREv will still work but the `cv2` module (used in `geo_utils.pansharpen()`) will not work. ')

if not find_loader('cv2'):
    _warnings.warn('\n*tensorflow* library is missing. storm-ASPIREv will still work but not for vegetation inference ')








