import time
from functools import lru_cache
from typing import Optional
# import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
from tiktoken import Encoding
from sfn_llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage
from sfn_llm_client.utils.consts import PROMPT_KEY
from sfn_llm_client.llm_cost_calculation.openai_cost_calculation import openai_cost_calculation
# import aiohttp
from sfn_llm_client.utils.logging import setup_logger
from sfn_llm_client.utils.retry_with import retry_with

class OpenAILangchainClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        # openai.api_key = self._api_key
        # self._client = openai
        # self._client = None
        self.logger, _ = setup_logger(logger_name="OpenAILangchainClient")
        # self._session = aiohttp.ClientSession()

    def _get_or_create_client(self, model: str) -> ChatOpenAI:
        if self._client is None or self._current_model != model:
            self._client = ChatOpenAI(
                api_key=self._api_key,
                model=model,
                temperature=0,
                max_tokens=None,
                timeout=None
            )
            self._current_model = model
        return self._client

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
        # self._get_or_create_client(model)
        client = ChatOpenAI(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        completions = client.invoke(messages)
        print("completions:" , completions)

        # Check if response is empty
        if not completions or not completions.content:
            raise ValueError("Received empty response from the openai llm")

        token_usage = completions.response_metadata["token_usage"]
        prompt_tokens = token_usage['prompt_tokens']
        completion_tokens = token_usage['completion_tokens']
        
        token_cost_summary = openai_cost_calculation(
            prompt_tokens,
            completion_tokens,
            model,
        )
        return completions, token_cost_summary