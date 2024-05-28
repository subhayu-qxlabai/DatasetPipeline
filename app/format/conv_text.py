"""
This module provides the `ConvTextFormat` class for converting a dataset with one or multiple columns of conversational data to a standard conversational format having system, user, and assistant.

Example Usage:

```python
from datasets import Dataset
from dedup.conv_text import ConvTextConfig, ConvTextFormat

# Create a dataset
dataset = Dataset.from_csv('data.csv')

# Create a configuration object
config = ConvTextConfig(
    column='conversation_column',
    # Add any other configuration options here
)

# Create a ConvTextFormat object
formatter = ConvTextFormat(dataset, config)

# Perform the conversion
converted_dataset = formatter.format()

# Print the converted dataset
print(converted_dataset)
```
"""

from datasets import Dataset

from pydantic import Field

from .base import BaseFormat, BaseFormatConfig
from ..helpers import run_parallel_exec
from ..helpers.text_utils import TextUtils as tu
from ..helpers.call_openai import call_openai_api
from ..constants import MessageRole as Role, MessageField


class ConversationalTextFormatConfig(BaseFormatConfig):
    column: str | None = Field(default=None, description="Name of the column with the conversation. Defaults to `null`.")
    conv_template: str | None = Field(default=None, description="Template for the conversation. Templates have to contain `{user}` and `{assistant}`, optionally `{system}`. Defaults to `null`.")


class ConversationalTextFormat(BaseFormat):
    """Converts a dataset having columns with conversation between between 2-3 entities in text format to a standard conversational format having system, user and assistant."""

    def __init__(
        self,
        dataset: Dataset,
        config: ConversationalTextFormatConfig = ConversationalTextFormatConfig(),
    ):
        super().__init__(dataset, config)
        self.config: ConversationalTextFormatConfig
        self._is_this_format = False

    @property
    def is_this_format(self) -> bool:
        return self._is_this_format

    @staticmethod
    def try_parsing_to_dict(template: str, text: str) -> dict[Role, str]:
        try:
            return {Role(k): v for k, v in tu.parse_to_dict(template, text).items()}
        except Exception as e:
            return {}

    def _get_text_conv_cols_openai(self):
        response = call_openai_api(
            messages=[
                {"role": "system", "content": "You are a smart assistant who can find a conversational dialogue in a given JSON/dictionary. You have to just give the key for which the value looks the most like a conversational dialogue or question answer or a back and forth between two/three parties."},
                {"role": "user", "content": "{'id': 'flan.450764',\n 'system_prompt': 'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.',\n 'question': 'Victims of domestic violence will have access to quality legal representation through a campaign undertaken by Idaho Supreme Court Chief Justice Linda Copple Trout and corporate leaders. \"Thousands of times a year, Idahoans are victims of domestic violence. The victims are often women and their children and they frequently have few resources with which to pursue their legal rights,\" Trout said Tuesday. \"This campaign helps fill that gap in legal services for women who need the help at a time when they are in crisis.\" The Idaho Partners for Justice Project has already secured pledges of more than $35,000 from law firms, attorneys, corporations and individuals. The goal is $100,000. The drive to pay for free legal services will continue for the next two months. The money goes to Idaho Legal Aid Services and the Idaho Volunteer Lawyers Program. Last year, more than 5,000 petitions were filed in Idaho for protection orders in domestic violence cases. More than 12,000 victims contacted shelters or crisis hotlines. Joining Trout in the announcement was Idaho Bar Association President Fred Hoopes of Idaho Falls and Ida-West Energy Co. Chief Executive Officer Randy Hill, members of the project\\'s executive committee. Also on hand were some women who were victims of such violence, but benefited from free legal services. Last year\\'s campaign generated enough money and resources to help more than 450 victims. The help ranged from representation in protection order hearings to legal assistance in divorce, visitation and child support cases. The donations are tax deductible. \\n\\nBased on the paragraph, does the response \"Police Chief Bob Ackerman and Victim\\'s Advocate Katherine Gonzalez\" correctly answer the question \"Who joined the Idaho Supreme Court Justice in making the announcement?\"?',\n 'response': 'Based on the paragraph, the response \"Police Chief Bob Ackerman and Victim\\'s Advocate Katherine Gonzalez\" does not correctly answer the question \"Who joined the Idaho Supreme Court Justice in making the announcement?\". The correct answer would be \"Idaho Bar Association President Fred Hoopes of Idaho Falls and Ida-West Energy Co. Chief Executive Officer Randy Hill, members of the project\\'s executive committee.\"',\n 'messages': '<<SYS>> You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps. <<SYS>>\\n### Instruction: Victims of domestic violence will have access to quality legal representation through a campaign undertaken by Idaho Supreme Court Chief Justice Linda Copple Trout and corporate leaders. \"Thousands of times a year, Idahoans are victims of domestic violence. The victims are often women and their children and they frequently have few resources with which to pursue their legal rights,\" Trout said Tuesday. \"This campaign helps fill that gap in legal services for women who need the help at a time when they are in crisis.\" The Idaho Partners for Justice Project has already secured pledges of more than $35,000 from law firms, attorneys, corporations and individuals. The goal is $100,000. The drive to pay for free legal services will continue for the next two months. The money goes to Idaho Legal Aid Services and the Idaho Volunteer Lawyers Program. Last year, more than 5,000 petitions were filed in Idaho for protection orders in domestic violence cases. More than 12,000 victims contacted shelters or crisis hotlines. Joining Trout in the announcement was Idaho Bar Association President Fred Hoopes of Idaho Falls and Ida-West Energy Co. Chief Executive Officer Randy Hill, members of the project\\'s executive committee. Also on hand were some women who were victims of such violence, but benefited from free legal services. Last year\\'s campaign generated enough money and resources to help more than 450 victims. The help ranged from representation in protection order hearings to legal assistance in divorce, visitation and child support cases. The donations are tax deductible. \\n\\nBased on the paragraph, does the response \"Police Chief Bob Ackerman and Victim\\'s Advocate Katherine Gonzalez\" correctly answer the question \"Who joined the Idaho Supreme Court Justice in making the announcement?\"?\\n### Response: Based on the paragraph, the response \"Police Chief Bob Ackerman and Victim\\'s Advocate Katherine Gonzalez\" does not correctly answer the question \"Who joined the Idaho Supreme Court Justice in making the announcement?\". The correct answer would be \"Idaho Bar Association President Fred Hoopes of Idaho Falls and Ida-West Energy Co. Chief Executive Officer Randy Hill, members of the project\\'s executive committee.\" <eos>'}"},
                {"role": "assistant", "content": "messages"},
                {"role": "user", "content": "{}"},
                {"role": "assistant", "content": "The given JSON/dictionary is empty. Hence, there is no key for conversational dialogue."},
                {"role": "user", "content": "{'text': 'Question: How does the process of oxidative decarboxylation of pyruvate contribute to the overall understanding of cellular respiration?\\n\\nAnswer: Oxidative decarboxylation of pyruvate is a crucial step in cellular respiration, as it generates energy in the form of ATP through the process of oxidative phosphorylation. This reaction occurs in the mitochondria and involves the transfer of electrons from pyruvate to coenzyme Q, which ultimately leads to the production of ATP. Additionally, the acetyl group produced during this process is used in the citric acid cycle, also known as the Krebs cycle or TCA cycle, where it is further metabolized to produce more ATP. Overall, the oxidative decarboxylation of pyruvate plays a central role in energy metabolism within cells.'}"},
                {"role": "assistant", "content": "text"},
                {"role": "user", "content": str(self.dict_repr)},
            ],
            n=2,
        )
        keys: list[str] = [
            x["message"]["content"]
            for x in response["choices"]
            if x["message"]["content"] in self.dict_repr
        ]
        return keys

    @staticmethod
    def _create_messages_from_template(
        template: str, text: str
    ) -> list[dict[str, str]]:
        parsed_dict = tu.parse_to_dict(template, text)
        roles_ordered = (Role.SYSTEM.value, Role.USER.value, Role.ASSISTANT.value)
        parsed_dict = sorted(
            parsed_dict.items(), key=lambda item: roles_ordered.index(item[0])
        )
        return [
            {MessageField.ROLE.value: k, MessageField.CONTENT.value: v}
            for k, v in parsed_dict
        ]

    @staticmethod
    def _get_template_openai(text: str, column_name: str = None):
        response = call_openai_api(
            messages=[
                {"role": "system", "content": "You'll be given a text as input. You need to return a template of how to insert variable texts inside that template. \n\nThese are the variables: {system}, {user}, {assistant}\n\nYou should find patterns in the text that can be used to fill different variable texts for system, user and assistant. You only have to return the template placeholders in place of the actual text."},
                {"role": "user", "content": "Question: Considering the role of plants in animal nutrition, how do plants synthesize energy and store it within their structure?\n\nAnswer: Plants synthesize energy through the process of photosynthesis, using carbon dioxide from the air, water, and inorganic elements from the soil. Energy from sunlight is trapped and used in these synthetic processes. This energy is then stored as chemical energy within the plant, which animals can utilize for maintaining life and synthesizing their own body tissues."},
                {"role": "assistant", "content": "Question: {user}\n\nAnswer: {assistant}"},
                {"role": "user", "content": "<<SYS>> You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old. <<SYS>>\n### Instruction: Bosna Hersek (BH) Başbakanı Adnan Terziç, Avrupa haritasında kara bir delik kalması halinde, AB ile bütünleşme sürecinin küresel anlamda tarihi bir örnek teşkil edemeyeceğini söyledi.\n\nTranslate to English\n\nEnglish:\n### Response: Bosnia and Herzegovina (BH) Prime Minister Adnan Terzic said that if there is a black hole in the European map, the integration process with the European Union (EU) cannot be a historic example in a global sense. <eos>"},
                {"role": "assistant", "content": "<<SYS>> {system} <<SYS>>\n### Instruction: {user}\n### Response: {assistant} <eos>"},
                {"role": "user", "content": "<<SYS>> You are an AI assistant. You will be given a task. You must generate a detailed and long answer. <<SYS>> [INST]  Outline the Imhotep theory [/INST] This theory was the famous Imhotep, the renowned ruler of the ancient kingdom of Kush.  He was a great ruler and led his kingdom very well.  The word „Ihotek”, which means “defence”, was used during his reign, but Ihsotek might have been the first thing of its own."},
                {"role": "assistant", "content": "<<SYS>> {system} <<SYS>> [INST] {user} [/INST] {assistant}"},
                {"role": "user", "content": "system\n You are a smart AI assistant. \n\nuser\n What temperature should I dial down my water heater to? \n\nassistant\n If you turn the water heater on at a lower temperature, then it will begin to heat up on a smaller scale.  There is a zone in the water heater that just starts to bubble, and it’s at the upper critical temperature.  After at that point, you should call the water heater’s service provider, or they may know that you should turn it up to fight its owner at the lower critical temperature."},
                {"role": "assistant", "content": "system\n {system} \n\nuser\n {user} \n\nassistant\n {assistant}"},
                {"role": "user", "content": text},
            ],
            n=5,
        )
        response: list[str] = [x["message"]["content"] for x in response["choices"]]
        return response

    def _get_col_template_map(self):
        conv_col_keys = (
            [self.config.column]
            if self.config.column
            else self._get_text_conv_cols_openai()
        )
        col_template_map = run_parallel_exec(
            lambda x: (
                [self.config.conv_template]
                if self.config.conv_template and self.config.column in conv_col_keys
                else self._get_template_openai(x[1], x[0])
            ),
            [
                (k, v)
                for k, v in self.dict_repr.items()
                if isinstance(v, str) and k in conv_col_keys
            ],
        )
        col_template_map = {
            col: [
                (template, self.try_parsing_to_dict(template, val))
                for template in set(templates)
            ]
            for (col, val), templates in col_template_map
        }
        col_template_map: dict[str, str] = {
            col: (
                template.replace("{system}", "{user}")
                if set(template_dict) == {Role.SYSTEM, Role.ASSISTANT}
                else template
            )
            for col, (template_dicts) in col_template_map.items()
            for (template, template_dict) in template_dicts
            if template_dict
            and set(template_dict)
            in [
                {Role.SYSTEM, Role.USER, Role.ASSISTANT},
                {Role.USER, Role.ASSISTANT},
                {Role.SYSTEM, Role.ASSISTANT},
            ]
        }
        self._is_this_format = bool(col_template_map)
        return col_template_map

    def _format(self) -> Dataset:
        col_template_map = self._get_col_template_map()
        if not self.is_this_format:
            return self.dataset
        dataset = self.dataset.map(
            lambda row: {
                col: self._create_messages_from_template(template, row[col])
                for col, template in col_template_map.items()
            }
        )
        return dataset
