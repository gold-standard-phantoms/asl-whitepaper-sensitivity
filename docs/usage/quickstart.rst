Quickstart
==========

First install the package. Follow :doc:`installation` to set up a project and install ASLSENS first.

After installation the command line tool ``aslsens`` wil be made available. You can run::

    aslsens path/to/output_file.csv path/to/config_file.json

to run either a sensitivity or uncertainty analysis of the Whitepaper equation.

The config file is a JSON document detailing the type of analysis and the parameters that are to be
varied and their distributions. The configuration files have differences depending on whether a
sensitivity or uncertainty analysis is run.

In both cases they share the following fields:


:analysis_type: (string) Defines which analysis is run, either "uncertainty" or "sensitivity".
:parameters: (object) an object which defines the parameters and their distributions. Any omitted
  will be set to their defaults (see each analysis type for the default values), if empty then all
  parameters will be set to their default for the ``analysis_type``. Each parameter is an object
  which must have an entry "distribution", and then corresponding entries dependent on what
  distribution is chosen.


The following parameters are supported:

:label_efficiency: Number between 0 and 1 that defines the degree of labelling.
:t1_arterial_blood: The longitudinal relaxation time in seconds.
:lambda_blood_brain: The blood brain partition coefficient in ml/g.
:transit_time_scale: Scaling factor that the ground truth "transit_time" is multiplied by.
:t1_tissue_scale: Scaling factor that the ground truth "t1" is multiplied by.
:perfusion_rate_scale: Scaling factor that the ground truth "perfusion_rate" is multiplied by.
:desired_snr: The base image SNR. A value of 0 means no noise is added.

The following distributions are supported:

:gaussian: Values are randomly drawn from a normal distribution. Two additional parameters are
   required: "mean" and "sd", the mean and standard deviation of the distribution, respectively.
   By setting the standard deviation to 0.0 then the same value is selected each time.
:uniform: Values are randomly drawn uniformly over a defined range. Two additional
  parameters are required: "min" and "max", the minimum and maximum values of the range.
:linear: Values are linearly spaced over a range. Requires two additional parameter: ``"min"`` and
  ``"max"``, the minimum and maximum values of the range. 


Sensitivity analysis
--------------------

When running a sensitivity analysis, each parameter that is to be varied in the analysis must have
``distribution`` ``"linear"``, with an additional parameter ``"size"``, which is an integer that
specifies the number of values. The model will be run for all combinations of the parameters that
have a linear distribution, which is the product of all the ``"size"`` entries.

The default sensitivity analysis configuration is a two-level analysis
of the parameter ``lambda_blood_brain`` only, with low value ``0.8`` and high value ``1.0``. The
corresponding configuration file for this is:

.. code-block:: json

    {
        "analysis_type": "sensitivity",
        "parameters": {
            "label_efficiency": {
                "distribution": "linear",
                "min": 0.85,
                "max": 0.0,
            },
            "t1_arterial_blood": {
               "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "lambda_blood_brain": {
                "distribution": "linear",
                "min": 0.8,
                "max": 1.0,
                "size": 2
            },
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "t1_tissue_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "transit_time_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "desired_snr": {
                "distribution": "gaussian",
                "mean": 0.0,
                "sd": 0
            }
        }
    }


Uncertainty analysis
---------------------
The uncertainty analysis runs the model a specified number of times, sampling each parameter from
their defined distribution.  It requires two additional root-level parameters: 

:number_samples: the total number of times to run the model. On a i5-8400 CPU it takes approximately
 30 seconds to run the model once.
:random_seed: seed for the random number generator (uses default_rng from numpy.random)

Both of these parameters do not have default values and must be supplied if ``analysis_type`` is 
``"uncertainty"``.

The parameter settings are given below.  This configuration file will run the model once using the
mean values of each of the parameters (as the default is to have standard deviation of 0):

.. code-block:: json

    {
        "analysis_type": "uncertainty",
        "random_seed": 12345,
        "number_samples": 1,
        "parameters": {
            "label_efficiency": {
                "distribution": "gaussian",
                "mean": 0.85,
                "sd": 0.0
            },
            "t1_arterial_blood": {
                "distribution": "gaussian",
                "mean": 1.65,
                "sd": 0.0
            },
            "lambda_blood_brain": {
                "distribution": "gaussian",
                "mean": 0.9,
                "sd": 0.0
            },
            "desired_snr": {
                "distribution": "gaussian",
                "mean": 0.0,
                "sd": 0.0
            },
            "perfusion_rate_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "t1_tissue_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            },
            "transit_time_scale": {
                "distribution": "gaussian",
                "mean": 1.0,
                "sd": 0.0
            }
        }
    }