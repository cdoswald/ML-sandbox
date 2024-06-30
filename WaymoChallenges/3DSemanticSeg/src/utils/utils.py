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
    """Load TFRecord dataset."""
    return tf.data.TFRecordDataset(
        os.path.join(datadir, filename),
        compression_type="",
    )


def extract_frames_from_datafile(
    dataset: tf.data.Dataset,
    max_n_frames: Optional[int] = None,
) -> List[open_dataset.Frame]:
    """Extract frames (sequences) from TFRecord dataset."""
    # Validate max_n_frames arg
    if max_n_frames is not None:
        if (max_n_frames <= 0) or (not isinstance(max_n_frames, int)):
            raise ValueError(
                f"max_n_frames argument ({max_n_frames}) must be positive integer"
            )
    # Extract frames
    frames = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
        if (max_n_frames is not None) and (len(frames) >= max_n_frames):
            break
    return frames


def convert_range_image_to_tensor(
    range_image: open_dataset.dataset_pb2.MatrixFloat
) -> tf.Tensor:
    """Convert range image from protocol buffer MatrixFloat object
    to Tensorflow tensor object.
    
    Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb.
    """
    tensor = tf.convert_to_tensor(range_image.data)
    return tf.reshape(tensor, range_image.shape.dims)


# [print(x.name) for x in frame.DESCRIPTOR.fields]