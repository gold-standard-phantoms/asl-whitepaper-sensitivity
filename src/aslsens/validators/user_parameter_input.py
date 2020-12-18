""" User input validator. Used to initalise the model
All validation rules are contained within this file.
"""
import jsonschema

from asldro.validators.parameters import ValidationError, Parameter, ParameterValidator
from asldro.validators.user_parameter_input import (
    LAMBDA_BLOOD_BRAIN,
    T1_ARTERIAL_BLOOD,
    LABEL_EFFICIENCY,
    DESIRED_SNR,
)

# String constants
PERFUSION_RATE_SCALE = "perfusion_rate_scale"
T1_TISSUE_SCALE = "t1_tissue_scale"
TRANSIT_TIME_SCALE = "transit_time_scale"

# supported distribution types
DIST_GAUSSIAN = "gaussian"
DIST_UNIFORM = "uniform"
DIST_LINEAR = "linear"
SUPPORTED_PARAM_DISTS = [DIST_UNIFORM, DIST_GAUSSIAN, DIST_LINEAR]

# analysis type
UNCERTAINTY = "uncertainty"
SENSITIVITY = "sensitivity"


INPUT_SCHEMA = {
    "type": "object",
    "required": ["analysis_type"],
    "oneOf": [
        {
            "properties": {
                "analysis_type": {"type": "string", "enum": [UNCERTAINTY]},
                "number_samples": {"type": "number"},
                "random_seed": {"type": "number"},
                "parameters": {
                    "type": "object",
                    "patternProperties": {
                        f"[{LABEL_EFFICIENCY}]"
                        f"[{T1_ARTERIAL_BLOOD}]"
                        f"[{LAMBDA_BLOOD_BRAIN}]"
                        f"[{PERFUSION_RATE_SCALE}]"
                        f"[{T1_TISSUE_SCALE}]"
                        f"[{TRANSIT_TIME_SCALE}]"
                        f"[{DESIRED_SNR}]": {
                            "type": "object",
                            "required": ["distribution"],
                            "oneOf": [
                                {
                                    "properties": {
                                        "distribution": {
                                            "type": "string",
                                            "enum": [DIST_GAUSSIAN],
                                        },
                                        "mean": {"type": "number"},
                                        "sd": {"type": "number"},
                                    },
                                    "required": ["mean", "sd"],
                                },
                                {
                                    "properties": {
                                        "distribution": {
                                            "type": "string",
                                            "enum": [DIST_UNIFORM],
                                        },
                                        "min": {"type": "number"},
                                        "max": {"type": "number"},
                                    },
                                    "required": ["min", "max"],
                                },
                                {
                                    "properties": {
                                        "distribution": {
                                            "type": "string",
                                            "enum": [DIST_LINEAR],
                                        },
                                        "min": {"type": "number"},
                                        "max": {"type": "number"},
                                    },
                                    "required": ["min", "max"],
                                },
                            ],
                        },
                    },
                },
            },
            "required": [
                "parameters",
                "number_samples",
                "random_seed",
            ],
        },
        {
            "properties": {
                "analysis_type": {"type": "string", "enum": [SENSITIVITY]},
                "number_samples": {"type": "number"},
                "random_seed": {"type": "number"},
                "parameters": {
                    "type": "object",
                    "patternProperties": {
                        f"[{LABEL_EFFICIENCY}]"
                        f"[{T1_ARTERIAL_BLOOD}]"
                        f"[{LAMBDA_BLOOD_BRAIN}]"
                        f"[{PERFUSION_RATE_SCALE}]"
                        f"[{T1_TISSUE_SCALE}]"
                        f"[{TRANSIT_TIME_SCALE}]"
                        f"[{DESIRED_SNR}]": {
                            "type": "object",
                            "required": ["distribution"],
                            "oneOf": [
                                {
                                    "properties": {
                                        "distribution": {
                                            "type": "string",
                                            "enum": [DIST_GAUSSIAN],
                                        },
                                        "mean": {"type": "number"},
                                        "sd": {"type": "number", "enum": [0]},
                                    },
                                    "required": ["mean", "sd"],
                                },
                                {
                                    "properties": {
                                        "distribution": {
                                            "type": "string",
                                            "enum": [DIST_LINEAR],
                                        },
                                        "min": {"type": "number"},
                                        "max": {"type": "number"},
                                        "size": {"type": "number"},
                                    },
                                    "required": ["min", "max", "size"],
                                },
                            ],
                        },
                    },
                },
            },
            "required": [
                "parameters",
            ],
        },
    ],
}

USER_INPUT_VALIDATOR_UNCERTAINTY = ParameterValidator(
    parameters={
        LAMBDA_BLOOD_BRAIN: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 0.9, "sd": 0.0},
        ),
        T1_ARTERIAL_BLOOD: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.65, "sd": 0.0},
        ),
        LABEL_EFFICIENCY: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 0.85, "sd": 0.0},
        ),
        PERFUSION_RATE_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        T1_TISSUE_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        TRANSIT_TIME_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        TRANSIT_TIME_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        DESIRED_SNR: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        ),
    }
)

USER_INPUT_VALIDATOR_SENSITIVITY = ParameterValidator(
    parameters={
        LAMBDA_BLOOD_BRAIN: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "linear", "min": 0.8, "max": 1.0, "size": 2},
        ),
        T1_ARTERIAL_BLOOD: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.65, "sd": 0.0},
        ),
        LABEL_EFFICIENCY: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 0.85, "sd": 0.0},
        ),
        PERFUSION_RATE_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        T1_TISSUE_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        TRANSIT_TIME_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        TRANSIT_TIME_SCALE: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 1.0, "sd": 0.0},
        ),
        DESIRED_SNR: Parameter(
            validators=[],
            optional=True,
            default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        ),
    }
)


def validate_input_parameters(input_params: dict) -> dict:

    try:
        jsonschema.validate(instance=input_params, schema=INPUT_SCHEMA)
    except jsonschema.exceptions.ValidationError as ex:
        # Make the type of exception raised consistent
        raise ValidationError from ex

    # validate the parameters object
    if input_params["analysis_type"] == UNCERTAINTY:
        input_params["parameters"] = USER_INPUT_VALIDATOR_UNCERTAINTY.validate(
            input_params["parameters"]
        )
    elif input_params["analysis_type"] == SENSITIVITY:
        input_params["parameters"] = USER_INPUT_VALIDATOR_SENSITIVITY.validate(
            input_params["parameters"]
        )
    return input_params