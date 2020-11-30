"""tests for main.py"""
import pytest

from asl_sens.main import whitepaper_model


def test_whitepaper_model():
    asldro_params = {
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
        "label_efficiency": 0.85,
        "asl_context": "m0scan control label control label control label control label",
        "desired_snr": 1e10,
        "acq_matrix": [64, 64, 20],
        "label_duration": 1.8,
        "signal_time": 3.6,
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
