from snowflake.cortex import Complete
import time
from functools import lru_cache
from typing import Optional
import openai
import tiktoken
from tiktoken import Encoding
from llm_client.llm_api_client.base_llm_api_client import (
    BaseLLMAPIClient,
    LLMAPIClientConfig,
    ChatMessage,
)
from llm_client.consts import PROMPT_KEY
from llm_client.logging import setup_logger
from snowflake.snowpark import Session


def chat_completion(
    messages: list[ChatMessage],
    temperature: float = 0,
    max_tokens: int = 16,
    top_p: float = 1,
    model: Optional[str] = "arctic",
    retries: int = 3,
    retry_delay: float = 3.0,
    session: Optional[Session] = None,
    **kwargs,
) -> list[str]:

    completions = Complete(
        model,
        prompt=messages,
        options={"temperature": temperature, "guardrails": False},
        session=session,
    )
    
    # Calculate token consumption
    # token_consumption_dict = snowflake_arctic_cost_calculation(
    #     completions.usage.prompt_tokens,
    #     completions.usage.completion_tokens,
    #     model=model,
    # )
    # return completions, token_consumption_dict
    print('completions--------------------------------------',completions)

    return completions, {}