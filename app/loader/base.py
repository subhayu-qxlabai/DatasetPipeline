from abc import ABC, abstractmethod

from datasets import Dataset
from ..models.base import BaseModel


class BaseConfig(BaseModel):
    path: str

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
    def load(self) -> Dataset:
        """
        Method that loads a Dataset.

        :return: A Dataset object.
        :rtype: Dataset
        """
        pass
