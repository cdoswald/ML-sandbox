"""Main execution"""

import logging
import os
import numpy as np
import pyarrow.parquet as pq

import polars as pl
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod
from waymo_open_dataset.utils import frame_utils

from utils import utils as utl
from utils import utils_constants as utl_c
from utils import utils_plotting as utl_p

if __name__ == '__main__':

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="main.log", encoding="utf-8", level=logging.DEBUG)
    logger.debug("Logger set-up successful")

    # Define constants
    LASER_NAME_MAP = dict(wod.LaserName.Name.items())
    CAMERA_NAME_MAP = dict(wod.CameraName.Name.items())
    RANGE_IMAGE_DIM_MAP = utl_c.get_range_image_final_dim_dict()
    SEG_IMAGE_DIM_MAP = utl_c.get_seg_image_final_dim_dict()

    # Import data files
    data_dir = "/workspace/hostfiles/3DSemanticSeg/data"
    data_files = os.listdir(data_dir)
    dataset = utl.load_datafile(data_dir, data_files[0]) #TODO: generalize

    # Extract frames
    frames = utl.extract_frames_from_datafile(dataset)

    # Parse range images
    range_images, camera_projections, seg_labels, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frames[24]) # TODO: generalize
    )

    ## range_images is a dictionary formatted: {laser_index: [return1, return2]}
    ## seg_labels is a dictionary formatted: {laser_index: [return1, return2]}
    range_image = utl.convert_range_image_to_tensor(
        range_images[wod.LaserName.TOP][0]
    )
    seg_image = utl.convert_range_image_to_tensor(
        seg_labels[wod.LaserName.TOP][0]
    )
    
    utl_p.plot_range_image_tensor(
        range_image,
        RANGE_IMAGE_DIM_MAP,
        invert_colormap=True,
    )

    print(seg_labels[wod.LaserName.TOP][0].shape.dims)

    

    utl_p.plot_range_image_tensor(
        seg_image,
        SEG_IMAGE_DIM_MAP,
        style_params={'cmap':'tab20'}
    )




    # for idx, frame in enumerate(frames):
    #     if frame.lasers[0].ri_return1.segmentation_label_compressed:
    #         break

    # frame = frames[24]

    # frame.lasers

    # [print(x.name) for x in frame.DESCRIPTOR.fields]