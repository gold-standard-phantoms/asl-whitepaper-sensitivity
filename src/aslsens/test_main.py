"""tests for main.py"""
import pytest
import os
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import numpy.testing
from numpy.random import default_rng

from aslsens.data.filepaths import SENS_ANALYSIS_TEST_DATA
from aslsens.main import (
    analyse_effects,
    get_random_variable,
    whitepaper_model,
    run_analysis,
)


def test_whitepaper_model():
    """Tests the whitepaper 'measurement' model - generation
    of ASL data using the DRO, then quantification using the
    white paper equation"""

    asldro_params = {
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
        "label_efficiency": 0.85,
        "asl_context": "m0scan control label control label control label control label",
        "desired_snr": 1e10,
        "acq_matrix": [64, 64, 20],
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

    results = whitepaper_model(asldro_params, quant_params)


def test_random_variable():
    """Checks that get_random_variable returns correct values"""
    seed = 12345
    rg = default_rng(seed=seed)
    random_spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 10.0}

    x = rg.normal(1000, 10.0)
    y = get_random_variable(random_spec, rng=seed)
    numpy.testing.assert_equal(x, y)

    random_spec = {"distribution": "uniform", "min": 12.3545, "max": 25.357}
    rg = default_rng(seed=seed)
    x = rg.uniform(12.3545, 25.357)
    y = get_random_variable(random_spec, rng=seed)
    numpy.testing.assert_equal(x, y)

    random_spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 0.0}
    y = get_random_variable(random_spec, rng=seed)
    numpy.testing.assert_equal(1000.0, y)

    random_spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 10.0}
    rg = default_rng(seed=seed)
    x = rg.normal(1000, 10.0, 1000)
    y = get_random_variable(random_spec, 1000, rng=seed)
    numpy.testing.assert_equal(x, y)

    # repeat, this time passing the random number generator
    random_spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 10.0}
    rg = default_rng(seed=seed)
    x = rg.normal(1000, 10.0, 1000)
    rg = default_rng(seed=seed)
    y = get_random_variable(random_spec, 1000, rng=seed)
    numpy.testing.assert_equal(x, y)

    # test 'linear' distribution
    random_spec = {"distribution": "linear", "min": 10, "max": 11, "size": 100}
    x = np.linspace(10, 11, 100)
    y = get_random_variable(random_spec)
    numpy.testing.assert_equal(x, y)

    # check that supplying size overrides the value in randomspec
    random_spec = {"distribution": "linear", "min": 10, "max": 11, "size": 100}
    x = np.linspace(10, 11, 200)
    y = get_random_variable(random_spec, 200)
    numpy.testing.assert_equal(x, y)


def test_analyse_effects():
    """Tests the analyse_effects function with a toy example with the results
    form a two-parameter sensitivity analysis."""

    # load in example results for two parameters:
    # label_efficiency: 0.8 and 0.9
    # t1_arterial_blood: 1.55 and 1.75
    results_df = pd.read_csv(SENS_ANALYSIS_TEST_DATA)
    results_df.pop("Unnamed: 0")

    effects = analyse_effects(results_df)
    # main effect label_efficiency for GM
    gm_values = results_df["calc GM mean"].values
    main_effect_label_efficiency = (np.sum(gm_values[2:4]) - np.sum(gm_values[:2])) / 2
    main_effect_t1_arterial_blood = (
        np.sum(gm_values[1::2]) - np.sum(gm_values[0::2])
    ) / 2
    numpy.testing.assert_almost_equal(
        effects["calc GM mean"].at["label_efficiency", "label_efficiency"],
        main_effect_label_efficiency,
    )
    numpy.testing.assert_almost_equal(
        effects["calc GM mean"].at["t1_arterial_blood", "t1_arterial_blood"],
        main_effect_t1_arterial_blood,
    )
    # there is only one two-way effect
    two_way_effect = (np.sum(gm_values[::3]) - np.sum(gm_values[1:3:1])) / 2
    numpy.testing.assert_almost_equal(
        effects["calc GM mean"].at["label_efficiency", "t1_arterial_blood"],
        two_way_effect,
    )


def test_run_analysis_uncertainty():
    "tests that the main analysis with an uncertainty analysis with 1 sample"
    input_params = {
        "parameters": {
            "label_efficiency": {"distribution": "gaussian", "mean": 0.85, "sd": 0.1},
            "t1_arterial_blood": {"distribution": "gaussian", "mean": 1.65, "sd": 0.1},
            "lambda_blood_brain": {"distribution": "uniform", "min": 0.8, "max": 1.0},
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0,
            },
            "t1_tissue_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.1},
            "transit_time_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.1},
            "desired_snr": {"distribution": "uniform", "min": 25, "max": 200},
        },
        "number_samples": 1,
        "random_seed": 0,
        "analysis_type": "uncertainty",
    }

    with TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, "output.csv")
        run_analysis(input_params, output_filename)


def test_run_analysis_sensitivity():
    "tests that the main analysis with a sensitivity analysis with 1 parameter"
    input_params = {
        "parameters": {
            "label_efficiency": {
                "distribution": "linear",
                "min": 0.8,
                "max": 0.9,
                "size": 2,
            },
        },
        "analysis_type": "sensitivity",
    }

    with TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, "output.csv")
        run_analysis(input_params, output_filename)
