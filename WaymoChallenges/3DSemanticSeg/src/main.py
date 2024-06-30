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

# Import data files
data_dir = "/workspace/hostfiles/3DSemanticSeg/data"
data_files = os.listdir(data_dir)
dataset = utils.load_datafile(data_dir, data_files[0]) #TODO: generalize

# Extract frames
frames = utils.extract_frames_from_datafile(dataset)

# Parse range images
range_images, camera_projections, seg_labels, range_image_top_pose = (
    frame_utils.parse_range_image_and_camera_projection(frames[0]) # TODO: generalize
)


 