Installation
=============

ASLDRO should be installed as a module. Once installed it can simply be run as a command-line tool.
For more information how to use a python package in this way please
see https://docs.python.org/3/installing/index.html

Python Version
---------------

We recommend using the latest version of Python. ASLSENS supports Python 3.7 and newer.

Dependencies
-------------

These dependencies will be installed automatically when installing ASLSENS

* `nibabel`_ provides read / write access to some common neuroimaging file formats
* `numpy`_ provides efficient calculations with arrays and matrices
* `jsonschema`_ provides an implementation of JSON Schema validation for Python
* `asldro`_ provides a digital reference object for Arterial Spin Labelling
* `pandas`_ a fast, powerful, flexible and easy to use open source data analysis
  and manipulation tool


.. _nibabel: https://nipy.org/nibabel/
.. _numpy: https://numpy.org/
.. _jsonschema: https://python-jsonschema.readthedocs.io/en/stable/
.. _pandas: https://pandas.pydata.org/
.. _asldro: https://pypi.org/project/asldro/


Set up
--------
Set up a virtual environment and install the module:

.. code-block:: sh

    $ pip install .