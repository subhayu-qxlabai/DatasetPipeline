"""
This module provides the `OutputAnalyzer` class, which is a subclass of `BaseAnalyzer` and is used to analyze the output of a dataset.

The `OutputAnalyzer` class has the following methods:

- `__init__(self, dataset: Dataset, config: BaseAnalyzerConfig = BaseAnalyzerConfig())`: Initializes the `OutputAnalyzer` object with a dataset and a configuration object. Calls the `__init__` method of the `BaseAnalyzer` class to set up the base analyzer.
- `_analyze(self) -> Dataset`: Performs the analysis on the dataset. In this case, it simply returns the original dataset without any modifications.

The `OutputAnalyzer` class is a specialized version of the `BaseAnalyzer` class that provides a basic analysis functionality by returning the input dataset unchanged.

```python
from dataset import Dataset
from config import BaseAnalyzerConfig
from output_analyzer import OutputAnalyzer

# Create a dataset object
dataset = Dataset()

# Create a configuration object
config = BaseAnalyzerConfig()

# Create an instance of the OutputAnalyzer class
output_analyzer = OutputAnalyzer(dataset, config)

# Perform the analysis
analyzed_dataset = output_analyzer.analyze()

# Print the analyzed dataset
print(analyzed_dataset)
```

"""

from datasets import Dataset

from .base import BaseAnalyzer, BaseAnalyzerConfig


class OutputAnalyzerConfig(BaseAnalyzerConfig):
    pass

class OutputAnalyzer(BaseAnalyzer):
    def __init__(self, dataset: Dataset, config: OutputAnalyzerConfig = OutputAnalyzerConfig()):
        super().__init__(dataset, config)

    def _analyze(self) -> Dataset:
        return self.dataset
