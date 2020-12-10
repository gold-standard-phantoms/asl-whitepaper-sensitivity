"""filepaths.py tests"""

import os

from asl_sens.data.filepaths import SENS_ANALYSIS_TEST_DATA


def test_file_paths_exist():
    assert os.path.isfile(SENS_ANALYSIS_TEST_DATA)