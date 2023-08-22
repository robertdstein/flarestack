""" This module provides the functionality to create a dataset index by instantiating a DatasetIndex object and importing all the available datasets. Each dataset, in turns, is expect to import `dataset_index` from this module, and adding its own information.
"""

import logging
from typing import List
from flarestack.data import Dataset

from flarestack.data.icecube import ps_v004_p02, nt_v005_p01, gfu_v002_p04

logger = logging.getLogger(__name__)


class DatasetIndex:
    """Class storing an index for available datasets"""

    def __init__(self) -> None:
        """constructor"""
        self.index: dict[str, Dataset] = dict()

    def add_dataset(self, dataset: Dataset) -> None:
        """adds a dataset to the index

        Args:
            name (str): assigned name of the dataset
            object (Dataset): dataset object
        """
        self.index[dataset.name] = dataset

    # supporting aliases does not play well with the typing consistency of dataset_index
    # tentatively comment out this logic, can be removed at a later stage
    # def add_alias(self, alias: str, name: str) -> None:
    #     """adds an alias for a dataset
    #
    #     Args:
    #         alias (str): alias name
    #         name (str): dataset name
    #     """
    #     dest = self.index[name]
    #     if isinstance(dest, Dataset):
    #         self.index[alias] = name
    #     else:
    #         logger.warning("f{name} is already an alias, aliasing {dest} instead.")
    #         self.index[alias] = dest

    def get_dataset(self, name: str) -> Dataset:
        """retrieve a dataset by name

        Args:
            name (str): dataset name

        Returns:
            Dataset: dataset
        """
        dest = self.index[name]
        if isinstance(dest, Dataset):
            return dest
        else:
            logger.info(f"{name} is an alias for {dest}")
            return self.index[dest]

    def get_dataset_list(self):
        """Get list of indexed datasets"""
        return self.index.keys()


dataset_index = DatasetIndex()

"""
Register datasets to index.
Currently, only the last release of each dataset is tracked.
"""
for dataset in [ps_v004_p02, nt_v005_p01, gfu_v002_p04]:
    dataset_index.add_dataset(dataset)
