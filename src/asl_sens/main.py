"""ASL Whitepaper Sensitivity Analysis"""

import sys
import os
import shutil
import json
from tempfile import TemporaryDirectory
from copy import deepcopy
import argparse
import pdb

import numpy as np

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
from asldro.cli import FileType

from asl_sens.filters.load_asl_bids_filter import LoadAslBidsFilter
from asl_sens.filters.asl_quantification_filter import AslQuantificationFilter
from asl_sens.filters.roi_statistics_filter import RoiStatisticsFilter

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
GROUND_TRUTH_MODULATE_PARAMS = ["perfusion_rate", "transit_time", "t1"]
DEFAULT_M0_TE = 0.01
DEFAULT_M0_TR = 10.0
DEFAULT_CL_TE = 0.01
DEFAULT_CL_TR = 5.0


def main():
    """Main function for the command line interface."""
    parser = argparse.ArgumentParser(
        description="""ASL Whitepaper sensitivity analys using the
        ASLDRO digital reference object to generate synthetic ASL data with which
        the perfusion rate is calculated using the ASL Whitepaper
        single-subtraction equation""",
    )
    parser.add_argument(
        "output",
        help="path and filename to save output to",
        type=FileType(extensions=["csv"]),
    )
    args = parser.parse_args()
    output_filename = args.output

    ## Vary perfusion_rate scaling
    num_prs = 21
    perfusion_rate_scale = np.linspace(0, 2, num_prs)

    # pre-allocate output array:
    # mean and SD for GM and WM, calculated and ground truth
    output_array = np.zeros((num_prs, 10))

    asldro_params = {
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
        "label_efficiency": 0.85,
        "asl_context": "m0scan control label control label control label control label",
        "desired_snr": 1e10,
        "acq_matrix": [197, 233, 189],
        "label_duration": 1.8,
        "signal_time": 3.6,
        "perfusion_rate": {
            "scale": 1.0,
        },
        "ground_truth": "hrgt_icbm_2009a_nls_3t",
    }

    quant_params = {
        "label_type": "pcasl",
        "model": "whitepaper",
        "label_duration": 1.8,
        "post_label_delay": asldro_params["signal_time"]
        - asldro_params["label_duration"],
        "label_efficiency": 0.85,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
    }

    results = []
    # loop over the perfusion rate
    for idx, prs in enumerate(perfusion_rate_scale):
        asldro_params["perfusion_rate"]["scale"] = prs
        results.append(whitepaper_model(asldro_params, quant_params))
        output_array[idx, 0] = prs * 60.0
        output_array[idx, 1] = prs * 20.0
        output_array[idx, 2] = results[idx]["calculated"]["GM"]["mean"]
        output_array[idx, 3] = results[idx]["calculated"]["GM"]["sd"]
        output_array[idx, 4] = results[idx]["calculated"]["WM"]["mean"]
        output_array[idx, 5] = results[idx]["calculated"]["WM"]["sd"]
        output_array[idx, 6] = results[idx]["ground_truth"]["GM"]["mean"]
        output_array[idx, 7] = results[idx]["ground_truth"]["GM"]["sd"]
        output_array[idx, 8] = results[idx]["ground_truth"]["WM"]["mean"]
        output_array[idx, 9] = results[idx]["ground_truth"]["WM"]["sd"]

    np.savetxt(
        output_filename,
        X=output_array,
        fmt="%.8f",
        delimiter=", ",
        header="input GM, input WM, calc GM mean,calc GM sd,calc WM mean,calc WM sd,gt GM mean,gt GM sd,gt WM mean,gt WM sd",
    )


def whitepaper_model(dro_params: dict, calc_params: dict) -> dict:
    """Function that generates synthetic ASL data using ASLDRO, then loads in the data
    and the  AslQuantificationFilter to calculate the perfusion rate.  Then, the resampled
    ground truth label map is used to calculate region statistics for the defined tissues.

    :param dro_params: Dictionary of DRO parameters - ground truth override and scaling parameters
        are removed and then this dictionary is merged with the ASL and ground truth image series
        parameters.
    :type dro_params: dict
    :param calc_params: Dictionary of parameters for the quantification model.
    :type calc_params: dict
    :return: dictionary containing ROI statistics for both calculated perfusion rate and the
        ground truth after resampling to the ASL acquisition resolution.
    :rtype: dict
    """
    # copy input dictionaries as they are modified
    asldro_params = dro_params.copy()
    quant_params = calc_params.copy()

    # check the inputs are valid
    # pop the values associated with the ground truth
    parameter_override = {}
    for param in GROUND_TRUTH_OVERRIDE_PARAMS:
        if asldro_params.get(param) is not None:
            parameter_override[param] = asldro_params.pop(param)

    ground_truth_modulate = {}
    for param in GROUND_TRUTH_MODULATE_PARAMS:
        if asldro_params.get(param) is not None:
            ground_truth_modulate[param] = asldro_params.pop(param)

    if asldro_params.get("ground_truth") is not None:
        ground_truth = asldro_params["ground_truth"]
    else:
        ground_truth = "hrgt_icbm_2009a_nls_3t"

    # construct the DRO input parameters
    input_params = validate_input_params(
        {
            "global_configuration": {
                "ground_truth": ground_truth,
                "image_override": {},
                "parameter_override": parameter_override,
                "ground_truth_modulate": ground_truth_modulate,
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

        calc_roi_stats_filter = RoiStatisticsFilter()
        calc_roi_stats_filter.add_parent_filter(
            asl_quantification_filter, io_map={"perfusion_rate": "image"}
        )
        calc_roi_stats_filter.add_input("label_map", seg_nifti_loader.outputs["image"])
        calc_roi_stats_filter.run()

        gt_roi_stats_filter = RoiStatisticsFilter()
        gt_roi_stats_filter.add_parent_filter(perf_nifti_loader)
        gt_roi_stats_filter.add_input("label_map", seg_nifti_loader.outputs["image"])
        gt_roi_stats_filter.run()

        output_dictionary = {}
        output_dictionary["calculated"] = calc_roi_stats_filter.outputs[
            RoiStatisticsFilter.KEY_REGION_STATS
        ]

        output_dictionary["ground_truth"] = gt_roi_stats_filter.outputs[
            RoiStatisticsFilter.KEY_REGION_STATS
        ]
        return output_dictionary


if __name__ == "__main__":
    main()