from copy import deepcopy
from jsonschema.validators import validate
import pytest
from asldro.validators.parameters import ValidationError
from aslsens.validators.user_parameter_input import validate_input_parameters

TEST_INPUT_UNCERTAINTY = {
    "parameters": {
        "label_efficiency": {"distribution": "gaussian", "mean": 0.85, "sd": 0.1},
        "t1_arterial_blood": {"distribution": "gaussian", "mean": 1.65, "sd": 0.1},
        "lambda_blood_brain": {"distribution": "uniform", "min": 0.8, "max": 1.0},
        "perfusion_rate_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        "t1_tissue_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        "transit_time_scale": {"distribution": "linear", "min": 0.9, "max": 1.1},
        "desired_snr": {"distribution": "uniform", "min": 25, "max": 200},
    },
    "number_samples": 10,
    "random_seed": 0,
    "analysis_type": "uncertainty",
}

TEST_INPUT_SENSITIVITY = {
    "parameters": {
        "label_efficiency": {
            "distribution": "linear",
            "min": 0.8,
            "max": 0.9,
            "size": 3,
        },
        "t1_arterial_blood": {
            "distribution": "linear",
            "min": 1.55,
            "max": 1.75,
            "size": 3,
        },
        "lambda_blood_brain": {
            "distribution": "linear",
            "min": 0.8,
            "max": 1.0,
            "size": 3,
        },
        "perfusion_rate_scale": {
            "distribution": "gaussian",
            "mean": 1.0,
            "sd": 0.0,
        },
        "t1_tissue_scale": {
            "distribution": "linear",
            "min": 0.9,
            "max": 1.1,
            "size": 3,
        },
        "transit_time_scale": {
            "distribution": "linear",
            "min": 0.9,
            "max": 1.1,
            "size": 3,
        },
        "desired_snr": {"distribution": "gaussian", "mean": 1e10, "sd": 0},
    },
    "analysis_type": "sensitivity",
}


def test_user_parameter_input_valid_input():
    """Checks that valid inputs pass"""
    validate_input_parameters(TEST_INPUT_UNCERTAINTY)
    validate_input_parameters(TEST_INPUT_SENSITIVITY)


def test_user_parameter_input_invalid_input():
    """checks that ValidationErrors are raised when input parameters
    are missing or incorrect"""

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d.pop("parameters")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d.pop("number_samples")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d.pop("random_seed")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d["parameters"]["label_efficiency"].pop("distribution")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d["parameters"]["label_efficiency"].pop("mean")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d["parameters"]["label_efficiency"].pop("sd")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d["parameters"]["label_efficiency"]["distribution"] = "uniform"
    d["parameters"]["label_efficiency"].pop("sd")
    d["parameters"]["label_efficiency"].pop("mean")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d["parameters"]["label_efficiency"]["min"] = 0
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d["parameters"]["label_efficiency"].pop("min")
    d["parameters"]["label_efficiency"]["max"] = 1
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d["parameters"]["label_efficiency"]["distribution"] = "linear"
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_SENSITIVITY)
    d.pop("analysis_type")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_SENSITIVITY)
    d.pop("parameters")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_SENSITIVITY)
    d["parameters"]["transit_time_scale"].pop("size")
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d["parameters"]["transit_time_scale"]["distribution"] = "uniform"
    with pytest.raises(ValidationError):
        validate_input_parameters(d)

    d = deepcopy(TEST_INPUT_SENSITIVITY)
    d["parameters"]["perfusion_rate_scale"]["sd"] = 1.0
    with pytest.raises(ValidationError):
        validate_input_parameters(d)


def test_user_parameter_input_defaults_uncertainty():
    """Tests that the default uncertainty parameters are correctly created"""

    params = validate_input_parameters(
        {
            "parameters": {},
            "number_samples": 10,
            "random_seed": 0,
            "analysis_type": "uncertainty",
        }
    )

    default_params = {
        "parameters": {
            "lambda_blood_brain": {"distribution": "gaussian", "mean": 0.9, "sd": 0.0},
            "t1_arterial_blood": {"distribution": "gaussian", "mean": 1.65, "sd": 0.0},
            "label_efficiency": {"distribution": "gaussian", "mean": 0.85, "sd": 0.0},
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0,
            },
            "t1_tissue_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "transit_time_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "desired_snr": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        },
        "number_samples": 10,
        "random_seed": 0,
        "analysis_type": "uncertainty",
    }

    assert params == default_params


def test_user_parameter_input_defaults_sensitivity():
    """Tests that the default sensitivity parameters are correctly created"""

    params = validate_input_parameters(
        {
            "parameters": {},
            "analysis_type": "sensitivity",
        }
    )

    default_params = {
        "parameters": {
            "lambda_blood_brain": {
                "distribution": "linear",
                "min": 0.8,
                "max": 1.0,
                "size": 2,
            },
            "t1_arterial_blood": {"distribution": "gaussian", "mean": 1.65, "sd": 0.0},
            "label_efficiency": {"distribution": "gaussian", "mean": 0.85, "sd": 0.0},
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0,
            },
            "t1_tissue_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "transit_time_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "desired_snr": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        },
        "analysis_type": "sensitivity",
    }

    assert params == default_params


def test_user_parameter_input_inserts_defaults():
    """Tests that default paramters are correctly inserted"""
    d = deepcopy(TEST_INPUT_UNCERTAINTY)
    d["parameters"].pop("lambda_blood_brain")
    d["parameters"].pop("transit_time_scale")

    validated_params = validate_input_parameters(d)
    assert validated_params == {
        "parameters": {
            "label_efficiency": {"distribution": "gaussian", "mean": 0.85, "sd": 0.1},
            "t1_arterial_blood": {"distribution": "gaussian", "mean": 1.65, "sd": 0.1},
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0,
            },
            "t1_tissue_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "lambda_blood_brain": {"distribution": "gaussian", "mean": 0.9, "sd": 0.0},
            "transit_time_scale": {"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
            "desired_snr": {"distribution": "uniform", "min": 25, "max": 200},
        },
        "number_samples": 10,
        "random_seed": 0,
        "analysis_type": "uncertainty",
    }
