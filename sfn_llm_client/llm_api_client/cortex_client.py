import json
# import re
from snowflake.cortex import Complete
from typing import Optional
from sfn_llm_client.llm_api_client.base_llm_api_client import (
    BaseLLMAPIClient,
    ChatMessage
)
from sfn_llm_client.utils.logging import setup_logger
from snowflake.snowpark import Session
from sfn_llm_client.llm_cost_calculation.snowflake_cortex_cost_calculation import snowflake_cortex_cost_calculation
from sfn_llm_client.utils.retry_with import retry_with
class CortexClient(BaseLLMAPIClient):
    def __init__(self):
        self.logger, _ = setup_logger(logger_name="SnowflakeCortex")

    @retry_with(retries=3, retry_delay=3.0, backoff=True)
    def chat_completion(
        self,
        messages: list[ChatMessage],
        temperature: float = 0,
        max_tokens: int = 16,
        top_p: float = 1,
        model: Optional[str] = "snowflake-arctic",
        retries: int = 3,
        retry_delay: float = 3.0,
        session: Optional[Session] = None,
        **kwargs,
    ) -> list[str]:
        self.logger.info('Started calling Cortex Complete API...')

        completions = Complete(
            model,
            prompt=messages,
            options={"max_tokens": max_tokens, "temperature": temperature, "guardrails": False},
            session=session,
        )

        self.logger.info(f"Received cortex {model}, Completions response...{completions}")
        
        # response_content = response['choices'][0]['messages']
        # pattern = re.compile(r'\{.*"text_response".*"mapping".*\}', re.DOTALL)
        # match = pattern.search(response_content)
        # if match:
        #     extracted_json = match.group(0)  # Extract the dictionary part
        # else:
        #     return {"text_response":"Null","mapping":{}}
        # try:
        #     response_content = json.loads(extracted_json)
        # except json.JSONDecodeError:
        #     self.error("Error: Failed to decode JSON")

        # Calculate token consumption

        token_cost_summary = snowflake_cortex_cost_calculation(
            response=completions,
            model=model
        )
        self.logger.info(f"After consumed token's cost calculation received token_cost_summary...{token_cost_summary}")

        return completions, token_cost_summary
    
    
    async def text_completion(self, *args, **kwargs):
        raise NotImplementedError("text_completion is not supported in CortexClient.")

    async def embedding(self, *args, **kwargs):
        raise NotImplementedError("embedding is not supported in CortexClient.")

    async def get_chat_tokens_count(self, *args, **kwargs):
        raise NotImplementedError("get_chat_tokens_count is not supported in CortexClient.")