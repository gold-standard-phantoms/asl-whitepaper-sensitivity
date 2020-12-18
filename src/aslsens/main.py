"""ASL Whitepaper Sensitivity Analysis"""

import logging
import pprint
import sys
import itertools
import os
import shutil
import json
import time
from tempfile import TemporaryDirectory
from copy import deepcopy
import argparse
import pdb

import pandas as pd
import numpy as np
from numpy.random import default_rng

from asldro.examples import run_full_pipeline
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.validators.user_parameter_input import (
    DESIRED_SNR,
    LABEL_EFFICIENCY,
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

from aslsens.filters.load_asl_bids_filter import LoadAslBidsFilter
from aslsens.filters.asl_quantification_filter import AslQuantificationFilter
from aslsens.filters.roi_statistics_filter import RoiStatisticsFilter
from aslsens.validators.user_parameter_input import (
    PERFUSION_RATE_SCALE,
    SENSITIVITY,
    T1_TISSUE_SCALE,
    TRANSIT_TIME_SCALE,
    UNCERTAINTY,
    validate_input_parameters,
)

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

logger = logging.getLogger(__name__)


def main():
    """Main function for the command line interface.

    Handles the command line interface
    Has two positional arguments:

    1. filename to save output to
    2. input parameter file

    Then runs run_analysis
    """
    parser = argparse.ArgumentParser(
        description="""ASL Whitepaper sensitivity analysis using the
        ASLDRO digital reference object to generate synthetic ASL data with which
        the perfusion rate is calculated using the ASL Whitepaper
        single-subtraction equation""",
    )
    parser.add_argument(
        "output",
        help="path and filename to save output to",
        type=FileType(extensions=["csv"]),
    )
    parser.add_argument(
        "params",
        type=FileType(extensions=["json"], should_exist=True),
        help="A path to a JSON file containing the input parameters (required)",
    )

    args = parser.parse_args()
    output_filename = args.output
    input_params = None

    # load in the input parameters
    if args.params is not None:
        with open(args.params) as json_file:
            input_params = json.load(json_file)

    run_analysis(input_params, output_filename)


def run_analysis(input_params: dict, output_filename):
    """Runs either an uncertainty or sensitivity analysis.
    For uncertainty, all the input parameters are sampled at
    random based on their input distributions.
    For sensitivity, a full-factorial combination of all
    the prescribed parameters is created.

    The model is then run for each set of parameters.  This
    comprises of creating ASL data using the ASLDRO, loading the
    BIDS output files in and then calculating CBF according to the Whitepaper
    single subtraction equation.

    :param input_params: dictionary of input parameters
    :type input_params: dict
    :param output_filename: path to file to save file as, must be a csv file
    :type output_filename: str or path

    """
    start_time = time.time()
    input_params = validate_input_parameters(input_params)

    asldro_params = {
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
        "label_efficiency": 0.85,
        "asl_context": "m0scan control label control label control label control label",
        "desired_snr": 1e10,
        "acq_matrix": [64, 64, 40],
        "label_duration": 1.8,
        "signal_time": 3.6,
        "perfusion_rate": {
            "scale": 1.0,
        },
        "transit_time": {
            "scale": 1.0,
        },
        "t1": {
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

    num_samples = 1
    model_parameters = {}
    ###############################################
    # construct the parameter arrays based on the type of analysis
    if input_params["analysis_type"] == UNCERTAINTY:
        # For uncertainty analysis then each parameter is randomly sampled from for each
        # time the model is run.

        num_samples = input_params["number_samples"]
        # create the randomly sampled variables using the random number generator.  All values are
        # pre-allocated which ensures that the results can be
        rng = default_rng(seed=input_params["random_seed"])

        # limit `perfusion_rate_scale` so it cannot be less than 0
        model_parameters["perfusion_rate_scale"] = np.clip(
            get_random_variable(
                input_params["parameters"][PERFUSION_RATE_SCALE], num_samples, rng
            ),
            0,
            None,
        )
        # limit `transit_time_scale` so it cannot be less than 0
        model_parameters["transit_time_scale"] = np.clip(
            get_random_variable(
                input_params["parameters"][TRANSIT_TIME_SCALE], num_samples, rng
            ),
            0,
            None,
        )
        # limit `t1_tissue_scale` so it cannot be less than 0
        model_parameters["t1_tissue_scale"] = np.clip(
            get_random_variable(
                input_params["parameters"][T1_TISSUE_SCALE], num_samples, rng
            ),
            0,
            None,
        )
        # ASLDRO has limits on allowable values for `lambda_blood_brain` between 0 and 1
        model_parameters["lambda_blood_brain"] = np.clip(
            get_random_variable(
                input_params["parameters"][LAMBDA_BLOOD_BRAIN], num_samples, rng
            ),
            0,
            1,
        )

        # limit `t1_arterial_blood` so it cannot be less than 0
        model_parameters["t1_arterial_blood"] = np.clip(
            get_random_variable(
                input_params["parameters"][T1_ARTERIAL_BLOOD], num_samples, rng
            ),
            0,
            None,
        )
        # ASLDRO has limits on allowable values for `label_efficiency` between 0 and 1
        model_parameters["label_efficiency"] = np.clip(
            get_random_variable(
                input_params["parameters"][LABEL_EFFICIENCY], num_samples, rng
            ),
            0,
            1,
        )
        # limit `desired_snr` so cannot be less than 0
        model_parameters["desired_snr"] = np.clip(
            get_random_variable(
                input_params["parameters"][DESIRED_SNR], num_samples, rng
            ),
            0,
            None,
        )

    elif input_params["analysis_type"] == SENSITIVITY:
        # if the analysis type is sensitivity, then a 'full factorial' design
        # will be used to select the number of values of each of the model inputs
        # and evaluates the model over every posisble combination of these.
        # In the input parameter file, a parameter that is to be included in this
        # analysis will have a distribution 'linear', and a parameter that is to be
        # kept fixed will have a distribution 'gaussian' and sd = 0.

        # create a list of numpy arrays with the values that will be used for each
        # instance the parameter is varied (or not in the case of fixed parameters)
        param_keys = input_params["parameters"].keys()
        values_for_each_parameter = [
            get_random_variable(input_params["parameters"][param])
            for param in param_keys
        ]
        combined_parameters = np.array(
            [i for i in itertools.product(*values_for_each_parameter)]
        )
        for i, param in enumerate(param_keys):
            model_parameters[param] = combined_parameters[:, i]

        num_samples = model_parameters["label_efficiency"].size

    # pre-allocate output array:
    # mean and SD for GM and WM, calculated and ground truth
    output_array = np.zeros((num_samples, 8))

    inputs_df = pd.DataFrame(model_parameters, dtype=float)

    results = []

    for idx in range(num_samples):

        logger.info(
            "####################\n Iteration No. %s of %s\n",
            pprint.pformat(idx),
            pprint.pformat(num_samples),
        )

        asldro_params["perfusion_rate"]["scale"] = model_parameters[
            "perfusion_rate_scale"
        ][idx]
        asldro_params["transit_time"]["scale"] = model_parameters["transit_time_scale"][
            idx
        ]
        asldro_params["t1"]["scale"] = model_parameters["t1_tissue_scale"][idx]
        asldro_params[LAMBDA_BLOOD_BRAIN] = model_parameters["lambda_blood_brain"][idx]
        asldro_params[T1_ARTERIAL_BLOOD] = model_parameters["t1_arterial_blood"][idx]
        asldro_params[LABEL_EFFICIENCY] = model_parameters["label_efficiency"][idx]
        asldro_params[DESIRED_SNR] = model_parameters["desired_snr"][idx]

        results.append(whitepaper_model(asldro_params, quant_params))
        output_array[idx, 0] = results[idx]["calculated"]["GM"]["mean"]
        output_array[idx, 1] = results[idx]["calculated"]["GM"]["sd"]
        output_array[idx, 2] = results[idx]["calculated"]["WM"]["mean"]
        output_array[idx, 3] = results[idx]["calculated"]["WM"]["sd"]
        output_array[idx, 4] = results[idx]["ground_truth"]["GM"]["mean"]
        output_array[idx, 5] = results[idx]["ground_truth"]["GM"]["sd"]
        output_array[idx, 6] = results[idx]["ground_truth"]["WM"]["mean"]
        output_array[idx, 7] = results[idx]["ground_truth"]["WM"]["sd"]

        # Create a pandas DataFrame to hold the results
        output_df = pd.DataFrame(
            output_array,
            columns=[
                "calc GM mean",
                "calc GM sd",
                "calc WM mean",
                "calc WM sd",
                "gt GM mean",
                "gt GM sd",
                "gt WM mean",
                "gt WM sd",
            ],
        )
        # concatenate with the input parameters
        results_df = pd.concat([inputs_df, output_df], axis=1)
        # save each time in the loop in case there is an error so to avoid data loss
        results_df.to_csv(output_filename, float_format="%.8f")
        logger.info("--- Elapsed %.2f seconds ---", (time.time() - start_time))

    # if a sensitivity analysis is run, analyse these results
    if input_params["analysis_type"] == SENSITIVITY:
        sens_analysis_results = analyse_effects(results_df)
        for key in sens_analysis_results:
            with open(output_filename, mode="a") as csv_file:
                csv_file.write("\n\n" + key + "\n")

            sens_analysis_results[key].to_csv(
                output_filename, float_format="%0.8f", mode="a"
            )
    logger.info("--- Total time %.2f seconds ---", (time.time() - start_time))


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

    logger.info("Launch ASLDRO with parameters:\n%s", pprint.pformat(input_params))

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


def get_random_variable(random_spec: dict, size=None, rng=None) -> np.ndarray:
    """Returns a randomly generated number based on the input dictionary's parameters


    :param random_spec: dictionary with fields

        * 'distribution': 'gaussian' for normal distribution, 'uniform' for a rectangular
        distribution, 'linear' for a linear sweep between the min and maximum values
        * 'mean' ('gaussian' only): mean value of normal distribution.
        * 'sd' ('gaussian' only): standard deviation value of normal distribution.
        * 'min' ('uniform' and 'linear'): minimum value of rectangular distribution.
        * 'max' ('uniform' and 'linear'): maximum value of rectangular distribution.
        * 'size' ('linear' only): the number of values to return

    :type random_spec: dict
    :param size: output shape, this will override 'size' in random_spec if supplied.
    :type size: int or tuple of ints
    :param rng: random number generator to use
    :return: the randomly generated number
    :rtype: numpy array
    """
    rng = default_rng(rng)

    if random_spec["distribution"] == "gaussian":
        out = rng.normal(random_spec["mean"], random_spec["sd"], size)
    elif random_spec["distribution"] == "uniform":
        out = rng.uniform(random_spec["min"], random_spec["max"], size)
    elif random_spec["distribution"] == "linear":
        if size is None:
            size = random_spec["size"]
        out = np.linspace(random_spec["min"], random_spec["max"], size)

    if isinstance(out, float):
        return np.array(out, ndmin=1)
    else:
        return out


def analyse_effects(results_df: pd.DataFrame) -> dict:
    """Analyses the main effect and two-way interactions from a 2-level
    sensitivity analysis

    Args:
        results_df (pd.DataFrame): the output from the sensitivity analysis

    Returns:
        dict: analysed results, with an interaction matrix for each output
        parameter.
    """

    # get the labels of the parameters: omit first column, last 8 are the results values
    all_parameter_labels = list(results_df.columns.values[0:-8])
    results_labels = list(results_df.columns.values[-8:-4])

    # determine which parameters are varying and so should be analysed
    sens_params = [
        param
        for param in all_parameter_labels
        if not all(results_df[param].values == results_df[param].values[0])
    ]

    encoded_levels = results_df[sens_params]
    for param in sens_params:
        values: np.array = results_df[param].values
        valmax = values.max()
        valmin = values.min()
        valavg = (valmax + valmin) / 2
        span = (valmax - valmin) / 2
        encoded_levels.loc[:, param] = encoded_levels.loc[:, param].map(
            lambda x: (x - valavg) / span
        )

    # main effects
    output = {}
    for result in results_labels:
        interactions_matrix = pd.DataFrame(
            index=sens_params,
            columns=sens_params,
        )
        for param in sens_params:
            effects = results_df.groupby(param)[result].mean()
            levels = encoded_levels[param].unique()
            interactions_matrix.loc[param, param] = sum(
                [level * effects.values[i] for i, level in enumerate(levels)]
            )
            # calculate the standard error for comparing the effects against
            interactions_matrix.loc["std_err", param] = results_df[result].std(
                ddof=0
            ) / np.sqrt(2 ** len(results_df[result]))
        output[result] = {}
        output[result] = interactions_matrix

    # two-way effects
    if len(sens_params) > 1:
        twoway_labels = list(itertools.combinations(sens_params, 2))
        for result in results_labels:
            for key in twoway_labels:
                keylist = list(key)
                effects = results_df.groupby(keylist)[result].mean()
                levels_i = encoded_levels[key[0]].unique()
                levels_j = encoded_levels[key[1]].unique()
                vals_i: np.ndarray = results_df[key[0]].unique()
                vals_j: np.ndarray = results_df[key[1]].unique()
                output[result].loc[key[0], key[1]] = sum(
                    [
                        lev_i * lev_j * effects.loc[vals_i[i], vals_j[j]] / 2
                        for i, lev_i in enumerate(levels_i)
                        for j, lev_j in enumerate(levels_j)
                    ]
                )
    return output


if __name__ == "__main__":
    main()