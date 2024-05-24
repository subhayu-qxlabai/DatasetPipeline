"""
This module provides the base classes and functions for analyzing datasets.

Classes:
    BaseConfig: The base configuration class for analyzers.
    BaseAnalyzer: The base analyzer class that defines the interface for dataset analyzers.

Methods:
    __init__(self, dataset: Dataset, config=BaseConfig()):
        Initializes the BaseAnalyzer object with the given dataset and configuration.

    has_config(self) -> bool:
        Property that returns a boolean indicating whether the analyzer has a configuration.

    name(self) -> str:
        Method that returns the name of the analyzer.

    _analyze(self) -> Dataset:
        Abstract method that returns a Dataset object with the analyzed fields.

    analyze(self) -> Dataset:
        Method that returns a Dataset object with the analyzed fields. If the analyzer has a configuration, it performs the analysis. Otherwise, it returns the original dataset.

    __or__(self, cls: type["BaseAnalyzer"]) -> BaseAnalyzer:
        Operator overload for chaining analyzers. Creates a new instance of the given class with the analyzed dataset and appends the name of the current analyzer to the list of analyzers.

Example Usage:
    from datasets import Dataset
    from .base import BaseAnalyzer, BaseConfig

    class MyConfig(BaseConfig):
        pass
        
    class MyAnalyzer(BaseAnalyzer):
        def _analyze(self) -> Dataset:
            # Analyze the dataset and return a new dataset with analyzed fields
            pass

    dataset = Dataset.from_pandas(...)
    config = MyConfig()
    analyzer = MyAnalyzer(dataset, config)
    analyzed_dataset = analyzer.analyze()


"""

from abc import ABC, abstractmethod

from datasets import Dataset
from ..models.base import BaseModel


class BaseConfig(BaseModel):
    pass

class BaseAnalyzer(ABC):
    def __init__(self, dataset: Dataset, config = BaseConfig()):
        """
        Initializes the BaseAnalyzer object with the given dataset.

        Parameters:
            dataset (Dataset): The dataset to be used for initialization.
            config (BaseConfig): The config of the analyzer.

        Returns:
            None
        """
        self.dataset = dataset
        self.config = config
        self.analyzers: list[str] = []
        """List of analyzers that the dataset is a type of."""
       
    @property 
    def has_config(self) -> bool:
        """
        Property that returns a boolean indicating whether the analyzer has a config.

        :return: A boolean indicating whether the analyzer has a config.
        :rtype: bool
        """
        return self.config is not None
    
    @property
    def name(self) -> str:
        """
        Method that returns the name of the analyzer.

        :return: The name of the analyzer.
        :rtype: str
        """
        return self.__class__.__name__.replace("Analyzer", "")
    
    @abstractmethod
    def _analyze(self) -> Dataset:
        """
        Method that returns a Dataset object with the analyzed fields.

        :return: A Dataset object with the analyzed fields.
        :rtype: Dataset
        """
        pass
    
    def analyze(self) -> Dataset:
        """
        Method that returns a Dataset object with the analyzed fields.

        :return: A Dataset object with the analyzed fields.
        :rtype: Dataset
        """
        if self.has_config:
            return self._analyze()
        return self.dataset

    def __or__(self, cls: type["BaseAnalyzer"]):
        instance = cls(self.analyze())
        instance.analyzers.append(self.name)
        instance.analyzers.extend(self.analyzers)
        return instance
    