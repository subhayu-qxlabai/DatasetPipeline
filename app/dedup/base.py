"""
This module provides classes and functions for deduplicating datasets.

Classes:
    BaseConfig: A configuration class for the deduplicator.
    BaseDedup: An abstract base class for deduplicating datasets.

Functions:
    get_empty_dataset: Returns an empty dataset with the given column names.

Usage Example:
    from base import BaseConfig, BaseDedup

    # Create a dataset
    dataset = Dataset.from_dict({'column1': [1, 2, 3], 'column2': [4, 5, 6]})

    # Create a deduplicator with a config
    config = BaseConfig()
    deduplicator = BaseDedup(dataset, config)

    # Perform deduplication
    deduplicated_dataset = deduplicator.dedup()

    # Access the deduplicated and duplicate datasets
    deduplicated = deduplicated_dataset['deduplicated']
    duplicates = deduplicated_dataset['duplicates']
"""

from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict

from ..models.base import BaseModel


def get_empty_dataset(columns: list[str]):
    return Dataset.from_dict({x: [] for x in columns})

class BaseConfig(BaseModel):
    pass

class BaseDedup(ABC):
    def __init__(self, dataset: Dataset, config = BaseConfig()):
        """
        Initializes the BaseDedup object with the given dataset.

        Parameters:
            dataset (Dataset): The dataset to be used for initialization.
            config (BaseConfig): The config of the deduplicator.

        Returns:
            None
        """
        self.dataset = dataset
        self.config = config
        self.deduplicators: list[str] = []
        """List of deduplicators that the dataset is a type of."""
       
    @property 
    def has_config(self) -> bool:
        """
        Property that returns a boolean indicating whether the deduplicator has a config.

        :return: A boolean indicating whether the deduplicator has a config.
        :rtype: bool
        """
        return self.config is not None
    
    @property
    def name(self) -> str:
        """
        Method that returns the name of the deduplicator.

        :return: The name of the deduplicator.
        :rtype: str
        """
        return self.__class__.__name__.replace("Dedup", "")
    
    @abstractmethod
    def _dedup(self) -> DatasetDict:
        """
        Method that returns a deduplicated DatasetDict object with keys: `deduplicated` and `duplicates`.

        :return: A deduplicated DatasetDict object with keys: `deduplicated` and `duplicates`.
        :rtype: DatasetDict
        """
        pass
    
    def dedup(self) -> DatasetDict:
        """
        Method that returns a deduplicated DatasetDict object with keys: `deduplicated` and `duplicates`.

        :return: A deduplicated DatasetDict object with keys: `deduplicated` and `duplicates`.
        :rtype: DatasetDict
        """
        if self.has_config:
            return self._dedup()
        return DatasetDict(
            deduplicated=self.dataset,
            duplicates=get_empty_dataset(self.dataset.column_names),
        )

    def __or__(self, cls: type["BaseDedup"]):
        instance = cls(self.dedup())
        instance.deduplicators.append(self.name)
        instance.deduplicators.extend(self.deduplicators)
        return instance
    