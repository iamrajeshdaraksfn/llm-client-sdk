import json
# import re
from snowflake.cortex import Complete
from typing import Optional
from llm_client.llm_api_client.base_llm_api_client import (
    ChatMessage
)
from llm_client.logging import setup_logger
from snowflake.snowpark import Session
from llm_client.llm_cost_calculation.snowflake_cortex_cost_calculation import snowflake_cortex_cost_calculation

class SnowflakeCortex:
    def __init__(self):
        self.logger, _ = setup_logger(logger_name="SnowflakeCortex")

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
            options={"temperature": temperature, "guardrails": False},
            session=session,
        )

        self.logger.info("Received cortex {model}, Completions response...{completions}")

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
        self.logger.info("After consumed token's cost calculation received token_cost_summary...{token_cost_summary}")

        return completions, token_cost_summary