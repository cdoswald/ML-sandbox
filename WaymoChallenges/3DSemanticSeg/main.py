import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="main.log", encoding="utf-8", level=logging.DEBUG)
logger.debug("Logger set-up successful")

import os
import numpy as np
import pyarrow.parquet as pq

import polars as pl
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

from utils import utils

data_dir = "data"
data_files = os.listdir(data_dir)

file = data_files[0]




for field in frame.DESCRIPTOR.fields:
    print(field.name)




range_images, camera_projections, seg_labels, range_image_top_pose = (
    frame_utils.parse_range_image_and_camera_projection(frame)
)
 
 