"""tests for main.py"""
import pytest
import numpy as np
import numpy.testing
from numpy.random import default_rng


from asl_sens.main import get_random_variable, whitepaper_model


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
