"""Utility functions for Waymo Open Dataset challenges."""

import os
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
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
    range_image: open_dataset.MatrixFloat,
) -> tf.Tensor:
    """Convert range image from protocol buffer MatrixFloat object
    to Tensorflow tensor object.

    Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb.
    """
    tensor = tf.convert_to_tensor(range_image.data)
    return tf.reshape(tensor, range_image.shape.dims)


def plot_range_image_tensor(
    range_image: tf.Tensor,
    dim_map: Dict[int, str],
    invert_colormap: bool = False,
    style_params: Optional[Dict] = None,
) -> None:
    """Plot tensor-formatted range image.
    
    Args
        range_image: range image formatted as tf.Tensor
        dim_map: dict mapping last dimension index of tensor to corresponding name
        invert_colormap: invert pixel intensities (light becomes dark and vice versa)
        style_params: dict mapping style param name to values
    
    Returns
        None
    """
    # Specify default style params
    config = {
        'figsize': (12, 8),
        'gridspec_kw': {'hspace': 0.3},
        'fontsize': 20,
        'pad_amt': 10,
        'subtitle_loc': 'left',
        'cmap': 'gray',
    }

    # Update style params
    if style_params is not None:
        for key, value in style_params.items():
            if key not in config:
                warnings.warn(f'Style param "{key}" is not currently supported')
            else:
                config[key] = style_params[key]

    # Invert pixel intensities
    if invert_colormap:
        range_image = tf.where(
            tf.greater_equal(range_image, 0),
            range_image,
            tf.ones_like(range_image) * 1e10
        )

    # Plot distance, intensity, and elongation
    fig, axes = plt.subplots(
        nrows=len(dim_map),
        figsize=config['figsize'],
        gridspec_kw=config['gridspec_kw'],
    )
    for idx, axes_name in dim_map.items():
        axes[idx].imshow(range_image[..., idx], cmap=config['cmap'], aspect='auto')
        axes[idx].set_title(
            axes_name,
            fontsize=config['fontsize'],
            pad=config['pad_amt'],
            loc=config['subtitle_loc'],
        )
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
