"""
This module defines the base classes for loading datasets.

Classes:
    BaseConfig: An empty base configuration model.
    BaseLoader: An abstract base class for loading datasets.

Usage:
    To use this module, you can create a subclass of `BaseLoader` and implement the `_load` method.
"""

from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict
from ..models.base import BaseModel


class BaseConfig(BaseModel):
    pass

class BaseLoader(ABC):
    def __init__(self, config: BaseConfig):
        """
        Initializes the BaseLoader object with the given dataset.

        Parameters:
            config (BaseConfig): The config of the loader.

        Returns:
            None
        """
        self.config = config
    
    @abstractmethod
    def _load(self) -> Dataset | DatasetDict:
        """
        Method that loads a Dataset or DatasetDict.

        :return: A Dataset object.
        :rtype: Dataset
        """
        pass
    
    def load(self) -> Dataset | DatasetDict | None:
        if self.config is None:
            return
        return self._load()
