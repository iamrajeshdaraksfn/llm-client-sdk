import json
# import re
from sfn_llm_client.llm_api_client.snowflake_cortex_complete_extended import complete, CompleteOptions
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
        model: Optional[str] = "claude-4-sonnet",
        temperature: float = 0,
        max_tokens: int = 1024,
        top_p: float = 0,
        session: Optional[Session] = None,
        **kwargs,
    ) -> list[str]:
        self.logger.info('Started calling Cortex Complete API...')

        options = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens,"top_p": top_p}.items() if v}
        if text_format := kwargs.get("text_format"): options["response_format"] = {"type": "json", "schema": text_format.model_json_schema()}
        if guardrails := kwargs.get("guardrails"): options["guardrails"] = guardrails
        completions, token_count = complete(model, prompt=messages, options=options, session=session)
        self.logger.info(f"Received cortex {model}, Completions response...{completions}")

        if text_format: completions = text_format.model_validate_json(completions)
        
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

        # token_cost_summary = snowflake_cortex_cost_calculation(
        #     response=token_count,
        #     model=model
        # )
        # cortex does not provide usage data
        token_cost_summary = {}
        self.logger.info(f"After consumed token's cost calculation received token_cost_summary...{token_cost_summary}")

        return completions, token_cost_summary
    
    
    async def text_completion(self, *args, **kwargs):
        raise NotImplementedError("text_completion is not supported in CortexClient.")

    async def embedding(self, *args, **kwargs):
        raise NotImplementedError("embedding is not supported in CortexClient.")

    async def get_chat_tokens_count(self, *args, **kwargs):
        raise NotImplementedError("get_chat_tokens_count is not supported in CortexClient.")