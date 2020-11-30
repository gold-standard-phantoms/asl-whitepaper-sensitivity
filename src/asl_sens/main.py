"""ASL Whitepaper Sensitivity Analysis"""

import os
import shutil
import json
from tempfile import TemporaryDirectory

import numpy as np
import nibabel as nib

from asldro.examples import run_full_pipeline
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.validators.user_parameter_input import (
    LAMBDA_BLOOD_BRAIN,
    T1_ARTERIAL_BLOOD,
    validate_input_params,
    ROT_X,
    ROT_Y,
    ROT_Z,
    TRANSL_X,
    TRANSL_Y,
    TRANSL_Z,
    ACQ_MATRIX,
    ECHO_TIME,
    REPETITION_TIME,
)

from asl_sens.filters.load_asl_bids_filter import LoadAslBidsFilter
from asl_sens.filters.asl_quantification_filter import AslQuantificationFilter

ARRAY_PARAMS = [
    ECHO_TIME,
    REPETITION_TIME,
    ROT_X,
    ROT_Y,
    ROT_Z,
    TRANSL_X,
    TRANSL_Y,
    TRANSL_Z,
]
GROUND_TRUTH_OVERRIDE_PARAMS = [LAMBDA_BLOOD_BRAIN, T1_ARTERIAL_BLOOD]
DEFAULT_M0_TE = 0.01
DEFAULT_M0_TR = 10.0
DEFAULT_CL_TE = 0.01
DEFAULT_CL_TR = 5.0


def whitepaper_model(asldro_params: dict, quant_params: dict) -> dict:

    # check the inputs are valid
    # pop the values associated with the ground truth
    parameter_override = {}
    for param in GROUND_TRUTH_OVERRIDE_PARAMS:
        if asldro_params.get(param) is not None:
            parameter_override[param] = asldro_params.pop(param)

    # construct the DRO input parameters
    input_params = validate_input_params(
        {
            "global_configuration": {
                "ground_truth": "hrgt_icbm_2009a_nls_3t",
                "image_override": {},
                "parameter_override": parameter_override,
            },
            "image_series": [
                {
                    "series_type": "asl",
                    "series_description": "asl image series",
                },
                {
                    "series_type": "ground_truth",
                    "series_description": "ground_truth image series",
                },
            ],
        }
    )

    # merge the new parameters
    asl_series_params = {
        **input_params["image_series"][0]["series_parameters"],
        **asldro_params,
    }
    timeseries_length = len(asl_series_params["asl_context"].split())
    # iterate through the array parameters, if they do not exist in asldro_params
    # then update the transformation parameters so they are the same length as asl_context
    for param in ARRAY_PARAMS:
        if param not in asldro_params.keys():
            if param == ECHO_TIME:
                asl_series_params[param] = [
                    DEFAULT_M0_TE if s == "m0scan" else DEFAULT_CL_TE
                    for s in asl_series_params["asl_context"].split()
                ]
            elif param == REPETITION_TIME:
                asl_series_params[param] = [
                    DEFAULT_M0_TR if s == "m0scan" else DEFAULT_CL_TR
                    for s in asl_series_params["asl_context"].split()
                ]
            else:
                asl_series_params[param] = [0.0] * timeseries_length

    input_params["image_series"][0]["series_parameters"] = asl_series_params
    ground_truth_series_params = input_params["image_series"][1]["series_parameters"]
    ground_truth_series_params[ACQ_MATRIX] = asl_series_params[ACQ_MATRIX]
    # assert 0

    # run the DRO, load in the results, calculate the CBF, analyse.
    with TemporaryDirectory() as temp_dir:
        archive_fn = os.path.join(temp_dir, "dro_out.zip")
        # run the DRO pipeline
        run_full_pipeline(input_params, archive_fn)
        # extract the DRO output to temp_dir
        shutil.unpack_archive(archive_fn, temp_dir)
        asl_fn = {
            LoadAslBidsFilter.KEY_IMAGE_FILENAME: os.path.join(
                temp_dir, "asl", "001_asl.nii.gz"
            ),
            LoadAslBidsFilter.KEY_SIDECAR_FILENAME: os.path.join(
                temp_dir, "asl", "001_asl.json"
            ),
            LoadAslBidsFilter.KEY_ASLCONTEXT_FILENAME: os.path.join(
                temp_dir, "asl", "001_aslcontext.tsv"
            ),
        }

        gt_seg_fn = os.path.join(
            temp_dir, "ground_truth", "002_ground_truth_seg_label.nii.gz"
        )
        gt_seg_sidecar_fn = os.path.join(
            temp_dir, "ground_truth", "002_ground_truth_seg_label.json"
        )

        gt_perf_fn = os.path.join(
            temp_dir, "ground_truth", "002_ground_truth_perfusion_rate.nii.gz"
        )
        # load in the asl bids files
        load_asl_bids_filter = LoadAslBidsFilter()
        load_asl_bids_filter.add_inputs(asl_fn)

        asl_quantification_filter = AslQuantificationFilter()
        asl_quantification_filter.add_parent_filter(load_asl_bids_filter)
        asl_quantification_filter.add_inputs(quant_params)

        asl_quantification_filter.run()
        calc_cbf = asl_quantification_filter.outputs["perfusion_rate"]

        # load in the ground truth segmentation
        seg_nifti_loader = NiftiLoaderFilter()
        seg_nifti_loader.add_input("filename", gt_seg_fn)
        seg_nifti_loader.run()

        with open(gt_seg_sidecar_fn, "r") as json_file:
            seg_nifti_loader.outputs["image"].metadata = json.load(json_file)
            json_file.close()

        # load in the ground truth perfusion rate
        perf_nifti_loader = NiftiLoaderFilter()
        perf_nifti_loader.add_input("filename", gt_perf_fn)
        perf_nifti_loader.run()

        label_map_data = seg_nifti_loader.outputs["image"].image
        label_regions = seg_nifti_loader.outputs["image"].metadata["LabelMap"]

        gt_cbf = perf_nifti_loader.outputs["image"].image

        output_dictionary = {}
        output_dictionary["calculated"] = {
            region: {
                "id": label_regions[region],
                "mean": np.mean(
                    calc_cbf.image[(label_map_data == label_regions[region])]
                ),
                "sd": np.std(calc_cbf.image[label_map_data == label_regions[region]]),
                "size": np.size(
                    calc_cbf.image[label_map_data == label_regions[region]]
                ),
            }
            for region in label_regions.keys()
        }

        output_dictionary["ground_truth"] = {
            region: {
                "id": label_regions[region],
                "mean": np.mean(gt_cbf[(label_map_data == label_regions[region])]),
                "sd": np.std(gt_cbf[label_map_data == label_regions[region]]),
                "size": np.size(gt_cbf[label_map_data == label_regions[region]]),
            }
            for region in label_regions.keys()
        }

        # shutil.move(asl_fn[LoadAslBidsFilter.KEY_IMAGE_FILENAME], output_directory)
        # shutil.move(gt_seg_fn, output_directory)
        # shutil.move(gt_perf_fn, output_directory)
        # nib.save(
        #    calc_cbf.nifti_image, os.path.join(output_directory, "calc_cbf.nii.gz")
        # )
        return output_dictionary
