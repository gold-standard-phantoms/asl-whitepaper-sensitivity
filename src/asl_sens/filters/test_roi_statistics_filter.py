"""Tests for roi_statistics_filter.py"""

from copy import deepcopy
import pytest

import numpy as np
import nibabel as nib
import numpy.testing

from asl_sens.filters.test_asl_quantification_filter import TEST_NIFTI_CON_ONES

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import NiftiImageContainer

from asl_sens.filters.roi_statistics_filter import RoiStatisticsFilter

from asl_sens.filters.test_asl_quantification_filter import (
    validate_filter_inputs,
)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = nib.Nifti2Image(np.ones(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
TEST_NIFTI_CON_ONES = NiftiImageContainer(nifti_img=TEST_NIFTI_ONES)
TEST_LABEL_MAP = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
    metadata={
        "LabelMap": {
            "region_0": 0,
            "region_1": 1,
            "region_2": 2,
        }
    },
)
TEST_LABEL_MAP_NO_LABELMAP = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
    metadata={
        "region_0": 0,
        "region_1": 1,
        "region_2": 2,
    },
)
TEST_LABEL_MAP_WRONG_KEYS = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
    metadata={
        "LabelMap": {
            0: 0,
            "region_1": 1,
            "region_2": 2,
        }
    },
)
TEST_LABEL_MAP_WRONG_VALUES = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
    metadata={
        "LabelMap": {
            "region_0": 0,
            "region_1": 1.8,
            "region_2": 2,
        }
    },
)

INPUT_VALIDATION_DICT = {
    "image": [False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1.0, "str"],
    "label_map": [
        False,
        TEST_LABEL_MAP,
        TEST_NIFTI_CON_ONES,
        TEST_LABEL_MAP_NO_LABELMAP,
        TEST_LABEL_MAP_WRONG_KEYS,
        TEST_LABEL_MAP_WRONG_VALUES,
        1.0,
        "str",
    ],
}
TEST_LABEL_MAP_WRONG_SHAPE = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(dataobj=np.ones((16, 16, 16)), affine=np.eye(4)),
    metadata={
        "LabelMap": {
            "region_0": 0,
            "region_1": 1,
            "region_2": 2,
        }
    },
)


def test_roi_statistics_filter_validate_inputs():
    """Checks that a FilterInputValidationError is raised when the inputs
    to the RoiStatisticsFilter are incorrect or missing"""

    # roi_statistics_filter = RoiStatisticsFilter()
    # roi_statistics_filter.add_input("image", TEST_NIFTI_CON_ONES)
    # roi_statistics_filter.add_input("label_map", TEST_LABEL_MAP)
    # roi_statistics_filter.run()

    validate_filter_inputs(RoiStatisticsFilter, INPUT_VALIDATION_DICT)


def test_roi_statistics_filter_with_mock_data():
    """Tests the RoiStatisticsFilter with some mock data"""

    image = TEST_NIFTI_CON_ONES.clone()
    label_map = TEST_LABEL_MAP.clone()

    # define regions and assign values
    # region 0
    image.image[:8, :, :] = np.random.normal(23.4, 5.0, (8, 32, 32))
    label_map.image[:8, :, :] = 0

    # region 1
    image.image[8:16, :, :] = 57.5
    label_map.image[8:16, :, :] = 1

    # region 2
    image.image[16:24, :, :] = -99.0
    label_map.image[16:24, :, :] = 2

    # region 3
    image.image[24:32, :, :] = 47.3
    label_map.image[24:32, :, :] = 3

    # create metadata entry, region_4 is defined here but not no voxels are assigned this value
    label_map.metadata["LabelMap"] = {
        "region_0": 0,
        "region_1": 1,
        "region_2": 2,
        "region_3": 3,
        "region_4": 4,
    }

    roi_statistics_filter = RoiStatisticsFilter()
    roi_statistics_filter.add_input("image", image)
    roi_statistics_filter.add_input("label_map", label_map)
    roi_statistics_filter.run()

    desired_region_stats = {
        "region_0": {
            "id": 0,
            "mean": np.mean(image.image[:8, :, :]),
            "sd": np.std(image.image[:8, :, :]),
            "size": np.size(image.image[:8, :, :]),
        },
        "region_1": {
            "id": 1,
            "mean": 57.5,
            "sd": 0.0,
            "size": np.size(image.image[8:16, :, :]),
        },
        "region_2": {
            "id": 2,
            "mean": -99.0,
            "sd": 0.0,
            "size": np.size(image.image[16:24, :, :]),
        },
        "region_3": {
            "id": 3,
            "mean": 47.3,
            "sd": 0.0,
            "size": np.size(image.image[24:32, :, :]),
        },
        "region_4": {
            "id": 4,
            "mean": 0,
            "sd": 0,
            "size": 0,
        },
    }
    region_stats = roi_statistics_filter.outputs["region_stats"]

    # compare each entry of region_stats with desired_region_stats
    for key in region_stats.keys():
        for subkey in region_stats[key].keys():
            numpy.testing.assert_allclose(
                region_stats[key][subkey], desired_region_stats[key][subkey], atol=1e-10
            )