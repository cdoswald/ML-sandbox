"""Utility functions for Waymo Open Dataset challenges."""

import os
from typing import List, Optional

import tensorflow
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset


def load_datafile(
    datadir: str,
    filename: str,
) -> tf.data.Dataset:
    """Load TFRecord dataset.
    
    Args
        datadir: name of data directory
        filename: name of .tfrecord file
    
    Returns
        tf.data.Dataset object
    """
    return tf.data.TFRecordDataset(
        os.path.join(datadir, filename),
        compression_type="",
    )


def extract_frames_from_datafile(
    dataset: tf.data.Dataset,
    max_n_frames: Optional[int] = None,
) -> List[open_dataset.Frame]:
    """Extract frames (sequences) from TFRecord dataset.
    
    Args
        dataset: TFRecord object
        max_n_frames: (default = None) max number of frames to extract
    
    Returns
        list of waymo_open_dataset.dataset_pb2.Frame objects
    """
    if (max_n_frames <= 0) or (not isinstance(max_n_frames, int)):
        raise ValueError(f"'max_n_frames' arg is '{max_n_frames}'--must be positive integer")
    frames = []
    for idx, data in enumerate(dataset):
        if idx >= max_n_frames:
            break
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
    return frames


# [print(x.name) for x in frame.DESCRIPTOR.fields]