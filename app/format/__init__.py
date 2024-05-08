from enum import Enum, auto
from functools import partial
from dataclasses import dataclass

from datasets import Dataset

from .base import BaseFormat, BaseConfig
from .sft import SFTFormat, SFTConfig, Role
from .merger import MergerFormat, MergerConfig, FieldConfig
from .conv import ConversationalFormat, ConvConfig
from .conv_text import ConversationalTextFormat, ConvTextConfig
from .dpo import DPOFormat, DPOConfig, DPOColumns
from .to_text import ToTextFormat, ToTextConfig, RoleConfig
from .output import OutputFormat, OutputConfig


class ChatFormat(Enum):
    ALPACA = auto()
    CHATML = auto()
    INST   = auto()

def get_format(text_format: ChatFormat) -> ToTextConfig:
    match text_format:
        case ChatFormat.ALPACA:
            return ToTextConfig(
                # system_key="### System: {system}",
                user_template="### Instruction: {user}",
                assistant_template="### Response: {assistant} <eos>",
                separator="\n",
            )
        case ChatFormat.CHATML:
            return ToTextConfig(
                user_template="user\n{user}",
                assistant_template="assistant\n{assistant}",
                separator="\n\n",
            )
        case ChatFormat.INST:
            return ToTextConfig(
                system_template="<<SYS>> {system} <<SYS>>",
                user_template="[INST] {user} [/INST]",
                assistant_template="{assistant}",
                separator=" ",
            )

class FormatConfig(BaseConfig):
    merger: MergerConfig | None = MergerConfig()
    sft: SFTConfig | None = SFTConfig()
    dpo: DPOConfig | None = DPOConfig()
    conv: ConvConfig | None = ConvConfig()
    conv_text: ConvTextConfig | None = None
    to_text: ToTextConfig | None = ToTextConfig()
    output: OutputConfig | None = OutputConfig()


@dataclass
class Format:
    """
    Formats a dataset into the standard format.
    
    Params:
        dataset (Dataset): The dataset to be formatted
        config (FormatConfig): The configuration for the format
    """
    dataset: Dataset
    config: FormatConfig | None = FormatConfig()
    
    def __post_init__(self):
        self.merger: type[MergerFormat] = partial(MergerFormat, config=self.config.merger)
        self.sft: type[SFTFormat] = partial(SFTFormat, config=self.config.sft)
        self.conv_text: type[ConversationalTextFormat] = partial(ConversationalTextFormat, config=self.config.conv_text)
        self.conv: type[ConversationalFormat] = partial(ConversationalFormat, config=self.config.conv)
        self.dpo: type[DPOFormat] = partial(DPOFormat, config=self.config.dpo)
        self.output: type[OutputFormat] = partial(OutputFormat, config=self.config.output)
        self.to_text: type[ToTextFormat] = partial(ToTextFormat, config=self.config.to_text)
    
    @property
    def _base_chain(self) -> DPOFormat:
        return (
            self.merger(self.dataset)
            | self.sft
            | self.conv_text # NOTE: This is an experimental format. Uncomment if you really need it
            | self.conv
            | self.dpo
        )
    
    def format(self, textualize: bool = False) -> Dataset:
        """
        Standardizes the dataset by applying a series of data transformations.

        Params:
            textualize (bool, optional): If True, the dataset is transformed into text format. Defaults to False.

        Returns:
            Dataset: The analyzed dataset.

        Note:
            - If `SFT` dataset, the returned dataset will have `messages` column.
            - If `DPO` dataset, the returned dataset will have `chosen` and `rejected` columns.
        """
        if self.config is None:
            return self.dataset
        chain = (self._base_chain | self.output)
        if not textualize:
            return chain.format()
        return (chain | self.to_text).format()
    