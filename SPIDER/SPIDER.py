# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The SPIDER dataset contains (human) lumbar spine magnetic resonance images 
(MRI) and segmentation masks described in the following paper:

van der Graaf, J.W., van Hooff, M.L., Buckens, C.F.M. et al. Lumbar spine 
segmentation in MR images: a dataset and a public benchmark. 
Sci Data 11, 264 (2024). https://doi.org/10.1038/s41597-024-03090-w

The dataset includes 447 sagittal T1 and T2 MRI series collected from 218 
patients across four hospitals. Segmentation masks indicating the vertebrae, 
intervertebral discs (IVDs), and spinal canal are also included. Segmentation 
masks were created manually by a medical trainee under the supervision of a 
medical imaging expert and an experienced musculoskeletal radiologist.

In addition to MR images and segmentation masks, additional metadata 
(e.g., scanner manufacturer, pixel bandwidth, etc.), limited patient 
characteristics (biological sex and age, when available), and radiological 
gradings indicating specific degenerative changes can be loaded with the 
corresponding image data.

HuggingFace repository: https://huggingface.co/datasets/cdoswald/SPIDER
""" 

# Import packages
import csv
import json
import os
import urllib.request
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

import datasets
import skimage
import SimpleITK as sitk

# Define functions
def import_csv_data(filepath: str) -> List[Dict[str, str]]:
    """Import all rows of CSV file."""
    results = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            results.append(line)
    return results

def subset_file_list(all_files: List[str], subset_ids: Set[int]):
    """Subset files pertaining to individuals in person_ids arg."""
    return ([
        file for file in all_files
        if any(str(id_val) == file.split('_')[0] for id_val in subset_ids)
    ])

def standardize_3D_image(
    image: np.ndarray,
    resize_shape: Tuple[int, int],
) -> np.ndarray:
    """Aligns dimensions of image to be (height, width, channels); resizes
    images to height/width values specified in resize_shape; and rescales 
    pixel values to Uint8."""
    # Align height, width, channel dims
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, axes=[1, 2, 0])
    # Resize image
    image = skimage.transform.resize(image, resize_shape)
    # Rescale to uint16 type (required for PyArrow and PIL)
    # image = skimage.img_as_ubyte(image)
    image = skimage.img_as_uint(image)
    return image

def standardize_3D_mask(
    mask: np.ndarray,
    resize_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Aligns dimensions of image to be (height, width, channels); resizes
    images to values specified in resize_shape using nearest neighbor interpolation; 
    and rescales pixel values to Uint8."""
    # Align height, width, channel dims
    if mask.shape[0] < mask.shape[2]:
        mask = np.transpose(mask, axes=[1, 2, 0])
    # Resize mask
    mask = skimage.transform.resize(
        mask,
        resize_shape,
        order=0,
        preserve_range=True,
        mode='constant',
        cval=0,
    )
    # Rescale to uint8 type (required for PyArrow and PIL)
    mask = skimage.img_as_ubyte(mask)
    return mask

# Define constants
MIN_IVD = 0
MAX_IVD = 9
DEFAULT_SCAN_TYPES = ['t1', 't2', 't2_SPACE']
DEFAULT_RESIZE = (512, 512)
DEMO_SUBSET_N = 10

_CITATION = """\
@misc{vandergraaf2023lumbar,
      title={Lumbar spine segmentation in MR images: a dataset and a public benchmark}, 
      author={Jasper W. van der Graaf and Miranda L. van Hooff and \
              Constantinus F. M. Buckens and Matthieu Rutten and \
              Job L. C. van Susante and Robert Jan Kroeze and \
              Marinus de Kleuver and Bram van Ginneken and Nikolas Lessmann},
      year={2023},
      eprint={2306.12217},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
"""

# Official description
_DESCRIPTION = """\
This is a large publicly available multi-center lumbar spine magnetic resonance \
imaging (MRI) dataset with reference segmentations of vertebrae, intervertebral \
discs (IVDs), and spinal canal. The dataset includes 447 sagittal T1 and T2 \
MRI series from 218 studies of 218 patients with a history of low back pain. \
The data was collected from four different hospitals. There is an additional \
hidden test set, not available here, used in the accompanying SPIDER challenge \
on spider.grand-challenge.org. We share this data to encourage wider \
participation and collaboration in the field of spine segmentation, and \
ultimately improve the diagnostic value of lumbar spine MRI.

This file also provides the biological sex for all patients and the age for \
the patients for which this was available. It also includes a number of \
scanner and acquisition parameters for each individual MRI study. The dataset \
also comes with radiological gradings found in a separate file for the \
following degenerative changes:

1.    Modic changes (type I, II or III)

2.    Upper and lower endplate changes / Schmorl nodes (binary)

3.    Spondylolisthesis (binary)

4.    Disc herniation (binary)

5.    Disc narrowing (binary)

6.    Disc bulging (binary)

7.    Pfirrman grade (grade 1 to 5). 

All radiological gradings are provided per IVD level.

Repository: https://zenodo.org/records/10159290
Paper: https://www.nature.com/articles/s41597-024-03090-w
"""

_HOMEPAGE = "https://zenodo.org/records/10159290"

_LICENSE = """Creative Commons Attribution 4.0 International License \
(https://creativecommons.org/licenses/by/4.0/legalcode)"""

_URLS = {
    "images":"https://zenodo.org/records/10159290/files/images.zip",
    "masks":"https://zenodo.org/records/10159290/files/masks.zip",
    "overview":"https://zenodo.org/records/10159290/files/overview.csv",
    "gradings":"https://zenodo.org/records/10159290/files/radiological_gradings.csv",
    "var_types": "https://huggingface.co/datasets/cdoswald/SPIDER/raw/main/textfiles/var_types.json",
}

class CustomBuilderConfig(datasets.BuilderConfig):
    
    def __init__(
        self,
        name: str = 'default',
        version: str = '0.0.0',
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence, Mapping]] = None,
        description: Optional[str] = None,
        scan_types: List[str] = DEFAULT_SCAN_TYPES,
        resize_shape: Tuple[int, int, int] = DEFAULT_RESIZE,
        shuffle: bool = True,
    ):
        super().__init__(name, version, data_dir, data_files, description)
        self.scan_types = self._validate_scan_types(scan_types)
        self.resize_shape = resize_shape
        self.shuffle = shuffle
        self.var_types = self._import_var_types()
            
    def _validate_scan_types(self, scan_types):
        for item in scan_types:
            if item not in ['t1', 't2', 't2_SPACE']:
                raise ValueError(
                    'Scan type "{item}" not recognized as valid scan type.\
                    Verify scan type argument.'
                )
        return scan_types
    
    def _import_var_types(self):
        """Import variable types from HuggingFace repository subfolder."""
        with urllib.request.urlopen(_URLS['var_types']) as url:
            var_types = json.load(url)
        return var_types


class SPIDER(datasets.GeneratorBasedBuilder):
    """Resized/rescaled 3-dimensional volumetric arrays of lumbar spine MRIs \
    with corresponding scanner/patient metadata and radiological gradings."""

    # Class attributes
    DEFAULT_WRITER_BATCH_SIZE = 16 # PyArrow default is too large for image data
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = CustomBuilderConfig
    BUILDER_CONFIGS = [
        CustomBuilderConfig(
            name="default",
            description="Load the full dataset",
        ),
        CustomBuilderConfig(
            name="demo",
            description="Generate 10 examples for demonstration",
        )
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        """Specify datasets.DatasetInfo object containing information and typing
        for the dataset."""
        
        features = datasets.Features({
            "patient_id": datasets.Value("string"),
            "scan_type": datasets.Value("string"),
            # "image": datasets.Array3D(shape=self.config.resize_shape, dtype='uint8'),
            # "mask": datasets.Array3D(shape=self.config.resize_shape, dtype='uint8'),
            "image": datasets.Sequence(datasets.Image()),
            "mask": datasets.Sequence(datasets.Image()),
            "image_path": datasets.Value("string"),
            "mask_path": datasets.Value("string"),
            "metadata": {
                k:datasets.Value(v) for k,v in 
                self.config.var_types['metadata'].items()
            },
            "rad_gradings": {
                "IVD label": datasets.Sequence(datasets.Value("string")),
                "Modic": datasets.Sequence(datasets.Value("string")),
                "UP endplate": datasets.Sequence(datasets.Value("string")),
                "LOW endplate": datasets.Sequence(datasets.Value("string")),
                "Spondylolisthesis": datasets.Sequence(datasets.Value("string")),
                "Disc herniation": datasets.Sequence(datasets.Value("string")),
                "Disc narrowing": datasets.Sequence(datasets.Value("string")),
                "Disc bulging": datasets.Sequence(datasets.Value("string")),
                "Pfirrman grade": datasets.Sequence(datasets.Value("string")),
            }
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self,
        dl_manager,
        validate_share: float = 0.2,
        test_share: float = 0.2,
        random_seed: int = 9999,
    ):
        """
        Download and extract data and define splits based on configuration.
        
        Args
            dl_manager: HuggingFace datasets download manager (automatically supplied)
            validate_share: float indicating share of data to use for validation;
                must be in range (0.0, 1.0); note that training share is 
                calculated as (1 - validate_share - test_share)
            test_share: float indicating share of data to use for testing;
                must be in range (0.0, 1.0); note that training share is 
                calculated as (1 - validate_share - test_share)
            random_seed: seed for random draws of train/validate/test patient ids
        """
        # Set constants
        train_share = (1.0 - validate_share - test_share)
        np.random.seed(int(random_seed))

        # Validate params
        if train_share <= 0.0:
            raise ValueError(
                f'Training share is calculated as (1 - validate_share - test_share) \
                and must be greater than 0. Current calculated value is \
                {round(train_share, 3)}. Adjust validate_share and/or \
                test_share parameters.'
            )
        if validate_share > 1.0 or validate_share < 0.0:
            raise ValueError(
                f'Validation share must be between (0, 1). Current value is \
                {validate_share}.'
            )
        if test_share > 1.0 or test_share < 0.0:
            raise ValueError(
                f'Testing share must be between (0, 1). Current value is \
                {test_share}.'
            )

        # Download images (returns dictionary to local cache)
        paths_dict = dl_manager.download_and_extract(_URLS)
                    
        # Get list of image and mask data files
        image_files = [
            file for file in os.listdir(os.path.join(paths_dict['images'], 'images'))
            if file.endswith('.mha')
        ]
        assert len(image_files) > 0, "No image files found--check directory path."
        
        mask_files = [
            file for file in os.listdir(os.path.join(paths_dict['masks'], 'masks'))
            if file.endswith('.mha')
        ]
        assert len(mask_files) > 0, "No mask files found--check directory path."
        
        # Filter image and mask data files based on scan types
        image_files = [
            file for file in image_files 
            if any(f'{scan_type}.mha' in file for scan_type in self.config.scan_types)
        ]

        mask_files = [
            file for file in mask_files 
            if any(f'{scan_type}.mha' in file for scan_type in self.config.scan_types)
        ]

        # Generate train/validate/test partitions of patient IDs
        patient_ids = np.unique([file.split('_')[0] for file in image_files])        
        partition = np.random.choice(
            ['train', 'dev', 'test'],
            p=[train_share, validate_share, test_share],
            size=len(patient_ids),
        )
        train_ids = set(patient_ids[partition == 'train'])
        validate_ids = set(patient_ids[partition == 'dev'])
        test_ids = set(patient_ids[partition == 'test'])
        assert len(train_ids.union(validate_ids, test_ids)) == len(patient_ids)

        # Subset train/validation/test partition images and mask files
        train_image_files = subset_file_list(image_files, train_ids)
        validate_image_files = subset_file_list(image_files, validate_ids)
        test_image_files = subset_file_list(image_files, test_ids)
        
        train_mask_files = subset_file_list(mask_files, train_ids)
        validate_mask_files = subset_file_list(mask_files, validate_ids)
        test_mask_files = subset_file_list(mask_files, test_ids)

        assert len(train_image_files) == len(train_mask_files)
        assert len(validate_image_files) == len(validate_mask_files)
        assert len(test_image_files) == len(test_mask_files)

        # Import patient/scanner data and radiological gradings data
        overview_data = import_csv_data(paths_dict['overview'])
        grades_data = import_csv_data(paths_dict['gradings'])

        # Convert overview data list of dicts to dict of dicts
        exclude_vars = ['new_file_name', 'subset'] # Original data only lists train and validate
        overview_dict = {}
        for item in overview_data:
            key = item['new_file_name']
            overview_dict[key] = {
                k:v for k,v in item.items() if k not in exclude_vars
            }
            overview_dict[key]['OrigSubset'] = item['subset'] # Change name to original subset

        # Convert overview data types
        cast_overview_dict = {}
        for scan_id, scan_metadata in overview_dict.items():
            cast_dict = {}
            for key, value in scan_metadata.items():
                if key in self.config.var_types['metadata']:
                    new_type = self.config.var_types['metadata'][key]
                    if new_type != "string":
                        cast_dict[key] = eval(f'np.{new_type}({value})')
                    else:
                        cast_dict[key] = str(value)
                else:
                    cast_dict[key] = value
            cast_overview_dict[scan_id] = cast_dict
        overview_dict = cast_overview_dict

        # Merge patient records for radiological gradings data
        grades_dict = {}
        for patient_id in patient_ids:
            patient_grades = [
                x for x in grades_data if x['Patient'] == str(patient_id)
            ]
            # Pad so that all patients have same number of IVD observations
            IVD_values = [x['IVD label'] for x in patient_grades]
            for i in range(MIN_IVD, MAX_IVD + 1):
                if str(i) not in IVD_values:
                    patient_grades.append({
                        "Patient": f"{patient_id}",
                        "IVD label": f"{i}",
                        "Modic": "",
                        "UP endplate": "",
                        "LOW endplate": "",
                        "Spondylolisthesis": "",
                        "Disc herniation": "",
                        "Disc narrowing": "",
                        "Disc bulging": "",
                        "Pfirrman grade": "",
                    })
            assert len(patient_grades) == (MAX_IVD - MIN_IVD + 1), "Radiological\
                gradings not padded correctly"
        
            # Convert to sequences
            df = (
                pd.DataFrame(patient_grades)
                .sort_values("IVD label")
                .reset_index(drop=True)
            )
            grades_dict[str(patient_id)] = {
                col:df[col].tolist() for col in df.columns
                if col not in ['Patient']
            }

        # DEMO configuration: subset first 10 examples
        if self.config.name == "demo":
            train_image_files = train_image_files[:DEMO_SUBSET_N]
            train_mask_files = train_mask_files[:DEMO_SUBSET_N]
            validate_image_files = validate_image_files[:DEMO_SUBSET_N]
            validate_mask_files = validate_mask_files[:DEMO_SUBSET_N]
            test_image_files = test_image_files[:DEMO_SUBSET_N]
            test_mask_files = test_mask_files[:DEMO_SUBSET_N]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "paths_dict": paths_dict,
                    "image_files": train_image_files,
                    "mask_files": train_mask_files,
                    "overview_dict": overview_dict,
                    "grades_dict": grades_dict,
                    "resize_shape": self.config.resize_shape,
                    "shuffle": self.config.shuffle,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "paths_dict": paths_dict,
                    "image_files": validate_image_files,
                    "mask_files": validate_mask_files,
                    "overview_dict": overview_dict,
                    "grades_dict": grades_dict,
                    "resize_shape": self.config.resize_shape,
                    "shuffle": self.config.shuffle,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "paths_dict": paths_dict,
                    "image_files": test_image_files,
                    "mask_files": test_mask_files,
                    "overview_dict": overview_dict,
                    "grades_dict": grades_dict,
                    "resize_shape": self.config.resize_shape,
                    "shuffle": self.config.shuffle,
                },
            ),
        ]

    def _generate_examples(
        self,
        paths_dict: Dict[str, str],
        image_files: List[str],
        mask_files: List[str],
        overview_dict: Dict,
        grades_dict: Dict,
        resize_shape: Tuple[int, int, int],
        shuffle: bool,
    ) -> Tuple[str, Dict]:
        """
        This method handles input defined in _split_generators to yield 
        (key, example) tuples from the dataset. The `key` is for legacy reasons 
        (tfds) and is not important in itself, but must be unique for each example.
        """
        # Shuffle order of patient scans
        # (note that only images need to be shuffled since masks and metadata
        # will be linked to the selected image)
        if shuffle:
            np.random.shuffle(image_files)

        ## Generate next example
        # ----------------------
        for idx, example in enumerate(image_files):

            # Extract linking data
            scan_id = example.replace('.mha', '')
            patient_id = scan_id.split('_')[0]
            scan_type = '_'.join(scan_id.split('_')[1:])

            # Load .mha image file
            image_path = os.path.join(paths_dict['images'], 'images', example)
            image = sitk.ReadImage(image_path)

            # Convert .mha image to original size numeric array
            image_array_original = sitk.GetArrayFromImage(image)
            
            # Convert .mha image to standardized numeric array
            image_array_standardized = standardize_3D_image(
                image_array_original,
                resize_shape,
            )
            
            # Split image array into sequence of 2D images
            split_len = image_array_standardized.shape[-1]
            images_seq = [
                np.squeeze(arr) for arr in np.split(
                    image_array_standardized,
                    split_len,
                    axis=-1,
                )
            ]

            # Load .mha mask file
            mask_path = os.path.join(paths_dict['masks'], 'masks', example)
            mask = sitk.ReadImage(mask_path)

            # Convert .mha mask to original size numeric array
            mask_array_original = sitk.GetArrayFromImage(mask)

            # Convert to Uint8 (existing range is [0,225], 
            # so all values should fit in Uint8)
            mask_array_standardized = np.array(mask_array_original, dtype='uint8')

            # Convert .mha mask to standardized numeric array
            mask_array_standardized = standardize_3D_mask(
                mask_array_standardized,
                resize_shape,
            )

            # Split mask array into sequence of 2D images
            split_len = mask_array_standardized.shape[-1]
            masks_seq = [
                np.squeeze(arr) for arr in np.split(
                    mask_array_standardized,
                    split_len,
                    axis=-1,
                )
            ]
    
            # Extract overview data corresponding to image
            image_overview = overview_dict[scan_id]

            # Extract patient radiological gradings corresponding to patient
            patient_grades_dict = grades_dict[patient_id]

            # Prepare example return dict
            return_dict = {
                'patient_id':patient_id,
                'scan_type':scan_type,
                'image':images_seq,
                'mask':masks_seq,
                'image_path':image_path, 
                'mask_path':mask_path, 
                'metadata':image_overview,
                'rad_gradings':patient_grades_dict,
            }

            # Yield example
            yield scan_id, return_dict
