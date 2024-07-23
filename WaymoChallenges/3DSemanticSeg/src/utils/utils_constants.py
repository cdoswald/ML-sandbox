"""Constant utility functions for Waymo Open Dataset challenges."""

import os
from typing import Dict, List, Optional, Tuple

from waymo_open_dataset import dataset_pb2 as wod


def get_range_image_final_dim_dict() -> Dict[int, str]:
    """Get dictionary mapping final dim of range images to signal type."""
    return {
        0: "Distance",
        1: "Intensity",
        2: "Elongation",
    }


def get_seg_image_final_dim_dict() -> Dict[int, str]:
    """Get dictionary mapping final dim of segmentation image to type."""
    return {
        0: "Instance ID",
        1: "Semantic Class",
    }
