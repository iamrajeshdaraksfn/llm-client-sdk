import time
from functools import lru_cache
from typing import Optional
import openai
import tiktoken
from tiktoken import Encoding
from sfn_llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage
from sfn_llm_client.utils.consts import PROMPT_KEY
from sfn_llm_client.llm_cost_calculation.openai_cost_calculation import openai_cost_calculation
# import aiohttp
from sfn_llm_client.utils.logging import setup_logger
from sfn_llm_client.utils.retry_with import retry_with

INPUT_KEY = "input"
MODEL_NAME_TO_TOKENS_PER_MESSAGE_AND_TOKENS_PER_NAME = {
    "gpt-3.5-turbo-0613": (3, 1),
    "gpt-3.5-turbo-16k-0613": (3, 1),
    "gpt-4-0314": (3, 1),
    "gpt-4-32k-0314": (3, 1),
    "gpt-4-0613": (3, 1),
    "gpt-4-32k-0613": (3, 1),
    "gpt-3.5-turbo-0301": (4, -1),
}


class OpenAIClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        openai.api_key = self._api_key
        self._client = openai
        self.logger, _ = setup_logger(logger_name="OpenAIClient")
        # self._session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session"""
        # await self._session.close()

    async def text_completion(self, prompt: str, model: Optional[str] = None, temperature: float = 0,
                              max_tokens: int = 16, top_p: float = 1, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        kwargs["top_p"] = top_p
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
        completions = await self._client.Completion.acreate(headers=self._headers, **kwargs, session=self._session)
        return [choice.text for choice in completions.choices]

    @retry_with(retries=3, retry_delay=3.0, backoff=True)
    def chat_completion(self, messages: list[ChatMessage], temperature: float = 0,
                        max_tokens: int = 16, top_p: float = 1, model: Optional[str] = None, 
                        retries: int = 3, retry_delay: float = 3.0, **kwargs) -> list[str]:
        """
        This method performs chat completion with OpenAI, and includes basic retry logic for handling
        exceptions or empty responses.

        :param retries: Number of retries in case of failure.
        :param retry_delay: Delay in seconds between retries.
        """
        self._set_model_in_kwargs(kwargs, model)
        messages = [
            message if isinstance(message, dict) else message.to_dict() 
            for message in messages
        ]
        completions = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Check if response is empty
        if not completions or not completions.choices:
            raise ValueError("Received empty response from the openai llm")

        token_cost_summary = openai_cost_calculation(
            completions.usage.prompt_tokens,
            completions.usage.completion_tokens,
            model=model,
        )
        return completions, token_cost_summary


    async def embedding(self, text: str, model: Optional[str] = None, **kwargs) -> list[float]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[INPUT_KEY] = text
        embeddings = await openai.Embedding.acreate(**kwargs, session=self._session)
        return embeddings.data[0].embedding

    async def get_tokens_count(self, text: str, model: Optional[str] = None, **kwargs) -> int:
        if model is None:
            model = self._default_model
        return len(self._get_relevant_tokeniser(model).encode(text))

    async def get_chat_tokens_count(self, messages: list[ChatMessage], model: Optional[str] = None) -> int:
        model = self._get_model_name_for_tokeniser(model)
        encoding = self._get_relevant_tokeniser(model)
        tokens_per_message, tokens_per_name = MODEL_NAME_TO_TOKENS_PER_MESSAGE_AND_TOKENS_PER_NAME[model]
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.content))
            num_tokens += len(encoding.encode(message.role.value))
            if message.name:
                num_tokens += len(encoding.encode(message.name))
                num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _get_model_name_for_tokeniser(self, model: Optional[str] = None) -> str:
        if model is None:
            model = self._default_model
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            return model
        elif model == "gpt-3.5-turbo-0301":
            return model
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning tokeniser assuming gpt-3.5-turbo-0613.")
            return "gpt-3.5-turbo-0613"
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning tokeniser assuming gpt-4-0613.")
            return "gpt-4-0613"
        else:
            raise NotImplementedError(
                f"""not implemented for model {model}. 
                See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    @staticmethod
    @lru_cache(maxsize=40)
    def _get_relevant_tokeniser(model: str) -> Encoding:
        return tiktoken.encoding_for_model(model)