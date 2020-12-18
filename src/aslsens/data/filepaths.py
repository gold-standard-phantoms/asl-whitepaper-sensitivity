"""Constants with data file paths"""

import os

# The data directory for the aslsens module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))

SENS_ANALYSIS_TEST_DATA = os.path.join(DATA_DIR, "sensitivity_analysis_test_data.csv")