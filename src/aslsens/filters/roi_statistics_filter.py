""" ROI Statistics Filter """

import numpy as np
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import BaseImageContainer
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)


class RoiStatisticsFilter(BaseFilter):
    """
    A filter that calculates statistics for regions in an image.

    **Inputs**
    Input Parameters are all keyword arguments for the :class:`RoiStatisticsFilter.add_input()`
    member function. They are also accessible via class constants, for example
    :class:`RoiStatisticsFilter.KEY_IMAGE`

    :param 'image': Input image to calculate region statistics on.
    :type 'image': BaseImageContainer
    :param 'label_map': Image with regions defined by assigning unsigned integer values to voxels
        that belong to the regions. Its ``metadata`` property contains a field ``'LabelMap'`` which
        has key/value pairs of the name/assigned value for each region.
    :type: 'label_map': BaseImageContainer

    An example ``'LabelMap'`` is shown below:

    .. code-block:: python

        {
            "region_1": 1,
            "region_2": 2,
            "region_3": 3,
        }

    If a region described in ``'LabelMap'`` has no assigned voxels in the ``label_map`` image
    the region will still be included in the output.

    **Outputs**

    :param 'region_stats': dictionary with an object for each region name as defined in
        ``'LabelMap'``, this then having fields:

        * 'id': the integer value assigned to the region.
        * 'mean': the mean value within the region.
        * 'sd': the standard deviation within the region.
        * 'size': the number of voxels in the region.

        For example:

        .. code-block:: python

            {
                "region_1": {
                    "id": 1,
                    "mean": 84.2,
                    "sd": 13.4,
                    "size": 2456,
                },
                "region_2": {
                    "id": 2,
                    "mean": 84.2,
                    "sd": 13.4,
                    "size": 2401,
                },
                "region_3": {
                    "id": 3,
                    "mean": 74.3,
                    "sd": 3.6,
                    "size": 6734
                },
            }

    """

    KEY_IMAGE = "image"
    KEY_LABEL_MAP = "label_map"
    KEY_REGION_STATS = "region_stats"

    def __init__(self):
        super().__init__(name="ROI Statistics")

    def _run(self):
        """calculates statistics in the input image based on the regions defined in
        the input label map
        """
        image_data: np.ndarray = self.inputs[self.KEY_IMAGE].image
        label_map_data: np.ndarray = self.inputs[self.KEY_LABEL_MAP].image
        label_regions = self.inputs[self.KEY_LABEL_MAP].metadata["LabelMap"]

        self.outputs[self.KEY_REGION_STATS] = {
            region: {
                "id": label_regions[region],
                "mean": np.mean(image_data[(label_map_data == label_regions[region])])
                if (label_map_data == label_regions[region]).any()
                else 0,
                "sd": np.std(image_data[label_map_data == label_regions[region]])
                if (label_map_data == label_regions[region]).any()
                else 0,
                "size": np.size(image_data[label_map_data == label_regions[region]])
                if (label_map_data == label_regions[region]).any()
                else 0,
            }
            for region in label_regions.keys()
        }

    def _validate_inputs(self):
        """Checks the inputs meet their validation criteria
        'image' must be derived from BaseImageContainer
        'label_map' must be derived from BaseImageContainer, have data type uint16,
        and have the metadata field 'LabelMap' that has key/value pairs that are str/int
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_LABEL_MAP: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        metadata_validator = ParameterValidator(
            parameters={"LabelMap": Parameter(validators=isinstance_validator(dict))}
        )
        metadata: dict = self.inputs[self.KEY_LABEL_MAP].metadata
        metadata_validator.validate(metadata, error_type=FilterInputValidationError)

        # Check that the metadata entry "LabelMap" has keys that are all strings, and values that
        # are all integers.
        if not all(isinstance(key, str) for key in metadata["LabelMap"].keys()):
            raise FilterInputValidationError(
                "all keys in metadata field 'LabelMap' in the input 'label_map must be of type str "
            )
        if not all(isinstance(val, int) for val in metadata["LabelMap"].values()):
            raise FilterInputValidationError(
                "all values in metadata field 'LabelMap' in the input 'label_map mus be of type int"
            )

        # Check that 'image' and 'label_map' have the same shape.
        if not (
            self.inputs[self.KEY_LABEL_MAP].shape == self.inputs[self.KEY_IMAGE].shape
        ):
            raise FilterInputValidationError(
                "inputs 'image' and 'label_map' must have the same shape"
            )
