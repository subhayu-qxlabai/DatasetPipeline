from abc import ABC, abstractmethod

from datasets import Dataset
from ..models.base import BaseModel


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
    def _dedup(self) -> Dataset:
        """
        Method that returns a deduplicated Dataset object.

        :return: A deduplicated Dataset object.
        :rtype: Dataset
        """
        pass
    
    def dedup(self) -> Dataset:
        """
        Method that returns a deduplicated Dataset object.

        :return: A deduplicated Dataset object.
        :rtype: Dataset
        """
        if self.has_config:
            return self._dedup()
        return self.dataset

    def __or__(self, cls: type["BaseDedup"]):
        instance = cls(self.dedup())
        instance.deduplicators.append(self.name)
        instance.deduplicators.extend(self.deduplicators)
        return instance
    