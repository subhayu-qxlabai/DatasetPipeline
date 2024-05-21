import json

from datasets import Dataset
from pydantic import model_validator,Field

from .base import BaseFormat, BaseConfig
from ..helpers.regex_dict import RegexDict
from ..helpers.call_openai import call_openai_api
from ..constants import MessageRole as Role, MessageField


PATTERN_ROLE_MAP = {
    "human.*": Role.USER,
    "question.*": Role.USER,
    "user.*": Role.USER,
    "dialogue.*": Role.USER,
    "input.*": Role.USER,
    "^prompt.*": Role.USER,
    "^instruction.*": Role.USER,
    "message_1": Role.USER,
    "source.*": Role.USER,
    
    "response.*": Role.ASSISTANT,
    "output.*": Role.ASSISTANT,
    "assistant.*": Role.ASSISTANT,
    "answer.*": Role.ASSISTANT,
    "summary.*": Role.ASSISTANT,
    "gpt.*": Role.ASSISTANT,
    "support.*": Role.ASSISTANT,
    "message_2": Role.ASSISTANT,
    "target.*": Role.ASSISTANT,
    
    "system.*": Role.SYSTEM,
    "instruction.*": Role.SYSTEM,
    "input.*": Role.SYSTEM,
}

class SFTConfig(BaseConfig):
    use_openai: bool = Field(default=False, description="Whether to use OpenAI to detect 'system', 'user' and 'assistant' columns. Defaults to 'false'")
    column_role_map: dict[str, Role|str] = Field(default=PATTERN_ROLE_MAP, description=f"Mapping between column names and role. Roles can be `user` and `assistant`, optionally `system`")

    @model_validator(mode="after")
    def validate_column_role_map(self):
        try:
            self.column_role_map = {
                k: v if isinstance(v, Role) else Role(v) 
                for k, v in self.column_role_map.items()
            }
        except Exception as e:
            raise ValueError(f'Values of `column_role_map` must be in the following: {", ".join(r.value for r in Role)}')
        return self


class SFTFormat(BaseFormat):
    def __init__(self, dataset: Dataset, config: SFTConfig = SFTConfig()):
        super().__init__(dataset, config)
        self.config: SFTConfig
        self.pattern_role_map = RegexDict(self.config.column_role_map)
        self.role_col_map = self._get_role_col_map()
    
    def _get_role_col_map(self) -> dict[Role, str]:
        data = self.dict_repr
        if not self.config.use_openai:
            role_col = []
            p_r_map: dict[str, Role] = self.pattern_role_map
            for col in list(data):
                key = p_r_map.get(col)
                if key is not None:
                    role_col.append((key, col))
                    p_r_map = RegexDict({
                        k: v for k, v in p_r_map.items() if v != key
                    })
            return dict(role_col)
        
        response = call_openai_api([
            {"role": "system", "content": 'You are supposed to understand a given data and find the keys that correspond to system, user and assistant roles. If you don\'t find a key, just put `null` as value. Your response should be in the following format: \n\n{"system": "key", "user": "key", "assistant": "key"}. \n\nRemember that there should be one `user` and one `assistant` type key. If you don\'t find such key return `{}`.'},
            {"role": "user", "content": '{"id": "flan.564327", "system_prompt": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.", "question": "Generate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One", "response": "Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One."}'},
            {"role": "assistant", "content": '{"system": "system_prompt", "user": "question", "assistant": "response"}'},
            {"role": "user", "content": '{"prompt": " How long will my leftovers keep refrigerated?", "response": "\\n\\n\\n It\\u2019s hard to say how long the leftovers will keep.  They might last for a few days in the refrigerator, but they\\u2019ll keep for a few weeks once left."}'},
            {"role": "assistant", "content": '{"system": null, "user": "prompt", "assistant": "response"}'},
            {"role": "user", "content": '{"system": "", "human": "You will be given a definition of a task first, then some input of the task.\\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\\n\\nAFC Ajax (amateurs)\'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\\nOutput:", "gpt": "[\\n  [\\"AFC Ajax (amateurs)\\", \\"has ground\\", \\"Sportpark De Toekomst\\"],\\n  [\\"Ajax Youth Academy\\", \\"plays at\\", \\"Sportpark De Toekomst\\"]\\n]"}'},
            {"role": "assistant", "content": '{"system": null, "user": "human", "assistant": "gpt"}'},
            {"role": "user", "content": '{"system": "", "question": "You will be given a definition of a task first, then some input of the task.\\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\\n\\nAFC Ajax (amateurs)\'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\\nOutput:", "chosen": "[\\n  [\\"AFC Ajax (amateurs)\\", \\"has ground\\", \\"Sportpark De Toekomst\\"],\\n  [\\"Ajax Youth Academy\\", \\"plays at\\", \\"Sportpark De Toekomst\\"]\\n]", "rejected": " Sure, I\'d be happy to help! Here are the RDF triplets for the input sentence:\\n\\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\\n\\nExplanation:\\n\\n* AFC Ajax (amateurs) is the subject of the first triplet, and hasGround is the predicate that describes the relationship between AFC Ajax (amateurs) and Sportpark De Toekomst.\\n* Ajax Youth Academy is the subject of the second triplet, and playsAt is the predicate that describes the relationship between Ajax Youth Academy and Sportpark De Toekomst.\\n\\nNote that there may be other possible RDF triplets that could be derived from the input sentence, but the above triplets capture the main relationships present in the sentence."}'},
            {"role": "assistant", "content": '{}'},
            {"role": "user", "content": json.dumps(data)}
        ])
        data: dict[str, str] = json.loads(response["choices"][0]["message"]["content"])
        return {Role(k): v for k, v in data.items()}
    
    @property
    def is_this_format(self) -> bool:
        if Role.ASSISTANT not in self.role_col_map:
            return False
        if Role.USER not in self.role_col_map:
            return False
        if len(self.role_col_map) >= 2:
            return True
        return False
    
    def _make_messages(self, data: dict[str, str]):
        return [
            {
                MessageField.ROLE.value: x.value, 
                MessageField.CONTENT.value: data[self.role_col_map[x]]
            } 
            for x in [Role.SYSTEM, Role.USER, Role.ASSISTANT] 
            if x in self.role_col_map
        ]

    def _format(self) -> Dataset:
        if not self.is_this_format:
            return self.dataset
        dataset = self.dataset.map(
            lambda x: {"messages": self._make_messages(x)}
        )
        self.messages_cols += ["messages"]
        return dataset
