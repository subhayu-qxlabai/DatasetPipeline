llm = {
    "v4": {
        "128k": {
            "150k": [
                {
                    "endpoint": "https://sweden-central-nlp-llm-instance-01.openai.azure.com/",
                    "key1": "11f4de001dca4e31ac5b124f71963c01",
                    "key2": "22ea99d60f844d59b7b7e511c5945b4a",
                    "region": "swedencentral",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
            ],
            "80k": [
                {
                    "endpoint": "https://canada-east-nlp-llm-instance-01.openai.azure.com/",
                    "key1": "c36fdaa4009b498d9e7056c58351fd98",
                    "key2": "115ab36028bd4ac28e4869d8dda164a8",
                    "region": "canadaeast",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
                {
                    "endpoint": "https://aus-east-nlp-llm-instance-01.openai.azure.com/",
                    "key1": "a54a2ee9ae054697b555d50f87964c07",
                    "key2": "f4815a369585405591f1aca8b8c0eecd",
                    "region": "australiaeast",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
                {
                    "endpoint": "https://uk-south-nlp-llm-instance-01.openai.azure.com/",
                    "key1": "d3d47affd52c487f908094d4f475ef93",
                    "key2": "59ef1b5b7f044841b14c03919cfb8ee7",
                    "region": "uksouth",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
                {
                    "endpoint": "https://fr-central-nlp-llm-instance-01.openai.azure.com/",
                    "key1": "142d309cf79f4ab0953227a1b6c4db6a",
                    "key2": "efd7f88d9f5b4468b4e9eaaf91564b61",
                    "region": "francecentral",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
                {
                    "endpoint": "https://east-us-2-llm-instance-01.openai.azure.com/",
                    "key1": "594a1155c8d948c2a8b535ab3adfa3bd",
                    "key2": "204722a1314f476f9620c5bafb5d5c20",
                    "region": "eastus2",
                    "model_id": "llm-chat-v4-128k-deploy-1",
                    "max_tokens": 4096,
                },
            ],
        },
    }
}

import copy
from time import sleep

last_gpt4_choosed_key = {"round": 0, "api": llm["v4"]["128k"]["150k"][0]}


def choosed_gpt4_key():
    # return last_gpt4_choosed_key
    global last_gpt4_choosed_key
    current_choosed_key = {}
    if last_gpt4_choosed_key["round"] == 0:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["150k"][0]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = 1
    elif last_gpt4_choosed_key["round"] == 1:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["150k"][0]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 2:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["150k"][0]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 3:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["150k"][0]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 4:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 4
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 5:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 4
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 6:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 4
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 7:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 4
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 8:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 4
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 9:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 9
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 10:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 9
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 11:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 9
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    elif last_gpt4_choosed_key["round"] == 12:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 9
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = last_gpt4_choosed_key["round"] + 1
    else:
        last_gpt4_choosed_key["api"] = llm["v4"]["128k"]["80k"][
            last_gpt4_choosed_key["round"] - 9
        ]
        current_choosed_key = copy.copy(last_gpt4_choosed_key)
        last_gpt4_choosed_key["round"] = 0
    return current_choosed_key


# requires openai==0.28.1
import openai
from typing import Callable

from .utils import get_openai_rate_limit_seconds
from .logger import LOGGER


def call_openai_api(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    n: int = 1,
    logger: Callable[[str, str], None] = LOGGER.log,
):
    """
    Call the OpenAI API to generate text based on the given messages.
    Args:
        messages (list[dict[str, str]]): A list of dictionaries representing the conversation messages.
        temperature (float, optional): The temperature parameter for text generation. Defaults to 0.7.
        n (int, optional): The number of text generation samples to generate. Defaults to 1.
        logger (Callable[[str, str], None], optional): A logger function to log debug and info messages. Defaults to LOGGER.log.
    Returns:
        The response from the OpenAI API.
    Raises:
        Exception: If the OpenAI API call fails or the response is filtered.
    Notes:
        - The OpenAI API is called using the selected GPT-4 key.
        - The OpenAI API version used is "2023-12-01-preview".
        - The OpenAI API key and base are set based on the selected GPT-4 key.
        - If the OpenAI API call fails due to rate limiting, the function retries after sleeping for the rate limit duration plus 60 seconds.
    Example:
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]
        response = call_openai_api(messages)
        print(response.choices[0].message.content)
    """
    
    llm_api = choosed_gpt4_key()["api"]
    # print(f"API: {llm_api}")

    openai_engine = llm_api["model_id"]
    openai_api_version = "2023-12-01-preview"
    openai.api_type = "azure"
    openai.api_key = llm_api["key1"]
    openai.api_base = llm_api["endpoint"]
    openai.api_version = openai_api_version

    try:
        response = openai.ChatCompletion.create(
            engine=openai_engine, messages=messages, temperature=temperature, n=n
        )
        return response
    except Exception as e:
        if "response was filtered" in str(e):
            logger("DEBUG", "Prompt was filtered.")
            return
        rate_limit_seconds = get_openai_rate_limit_seconds(str(e))
        seconds_to_sleep = rate_limit_seconds + 60
        logger("DEBUG", f"Got error: {e!r}")
        logger("INFO", f"OpenAI API rate limited, sleeping for {seconds_to_sleep} seconds...")
        sleep(seconds_to_sleep)
        return call_openai_api(messages=messages, temperature=temperature, n=n)


if __name__ == "__main__":
    print(
        call_openai_api(
            messages=[{"role": "user", "content": "Only give the answer. 1+2="}],
            temperature=0,
        )
        .choices[0]
        .message.content
    )
