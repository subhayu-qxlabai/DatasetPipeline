from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset
from ..models.base import BaseModel


class BaseConfig(BaseModel):
    pass

class BaseSaver(ABC):
    def __init__(self, dataset: Dataset, config: BaseConfig):
        """
        Initializes the BaseSaver object with the given dataset.

        Parameters:
            dataset (Dataset): The dataset to save.
            config (BaseConfig): The config of the saver.

        Returns:
            None
        """
        self.dataset = dataset
        self.config = config
    
    @abstractmethod
    def save(self) -> Path:
        """
        Method that returns a path to the saved Dataset.

        :return: A path to the saved Dataset.
        :rtype: Dataset
        """
        pass
