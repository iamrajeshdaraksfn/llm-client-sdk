from snowflake.snowpark import Session
from langchain_community.chat_models.snowflake import ChatSnowflakeCortex
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import model_validator
from typing import Dict
from sfn_llm_client.llm_api_client.base_llm_api_client import ChatMessage
from typing import Optional
from sfn_llm_client.utils.consts import PROMPT_KEY
from sfn_llm_client.llm_cost_calculation.snowflake_cortex_cost_calculation import snowflake_cortex_cost_calculation_langchain
from sfn_llm_client.llm_cost_calculation.openai_cost_calculation import openai_cost_calculation
from sfn_llm_client.utils.logging import setup_logger
from sfn_llm_client.utils.retry_with import retry_with

# Custom class that inherits from ChatSnowflakeCortex to override methods.
class CustomChatSnowflakeCortex(ChatSnowflakeCortex):
    """
    A custom ChatSnowflakeCortex class to demonstrate overriding the
    environment validation logic.
    """
    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Overrides the default pydantic environment validation.
        The original validator ensures a snowpark_session is present,
        which is a critical check we will maintain.
        """
        print("Executing custom override of validate_environment.")
        print("DEBUG: validate_environment() received values:", values)
        session = values.get("session")
        if session and isinstance(session, Session):
            print(f"Type of session: {type(session)}")
            print("DEBUG: Existing Snowpark session found. Skipping new session creation.")
            return values

        # If no session is provided, let the original parent validator handle the creation using environment variables.
        print("DEBUG: No session found. Using parent validation to create one.")
        return super().validate_environment(values)

class CortexLangchainClient():
    
    def __init__(self):
       
        self.logger, _ = setup_logger(logger_name="CortexLangchainClient")

    @retry_with(retries=3, retry_delay=3.0, backoff=True)
    def chat_completion(self,
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
        """
        Generates a response from Snowflake Cortex using the custom chat model.
        """
        client = CustomChatSnowflakeCortex(
            model=model,
            cortex_function="try_complete",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            session=session,
            snowflake_username="oauth_user",  # Dummy username to pass validation
        )
        try:
            cleaned_messages = self.convert_messages(messages)
            completions = client.invoke(cleaned_messages)
            print("completions: ", completions)

            # Check if response is empty
            if not completions or not completions.content:
                raise ValueError("Received empty response from the openai llm")

            print("completions:" , completions)

            token_cost_summary = snowflake_cortex_cost_calculation_langchain(
            response_metadata=completions.response_metadata,
            model=model
            )
            self.logger.info(f"After consumed token's cost calculation received token_cost_summary...{token_cost_summary}")

            return completions, token_cost_summary
        except Exception as e:
            print(f"Error during Snowflake Cortex generation: {e}")
            return f"Error: Could not process request with Snowflake Cortex. {e}", None
        
    def convert_messages(self, raw_messages: list[dict]) -> list:
        """
        Convert dict-based messages to LangChain message objects with cleaning.
        Supports 'system', 'user', and 'assistant' roles.
        """
        role_to_class = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage
        }

        converted = []
        for msg in raw_messages:
            role = msg.get("role", "").lower()
            content = self.clean_prompt(msg.get("content", ""))
            msg_class = role_to_class.get(role)
            if msg_class:
                converted.append(msg_class(content=content))
            else:
                print(f"Skipping unsupported role: {role}")
        return converted
    
    def clean_prompt(self, raw_prompt: str) -> str:
        """
        Clean prompt to avoid SQL syntax errors and ensure JSON compatibility.
        This approach focuses on escaping characters that would break SQL without
        interfering with JSON string integrity.
        """
        if not isinstance(raw_prompt, str):
            return ""
        cleaned_prompt = raw_prompt.rstrip("\n").splitlines()
        cleaned_prompt = " ".join(cleaned_prompt).replace("'", " ").replace('"', " ")
        return cleaned_prompt

    def get_langchain_llm(self, model="snowflake-arctic", temperature=0, max_tokens=16, top_p=1, session=None):
        """
        Returns a CustomChatSnowflakeCortex instance for agentic workflows.
        """
        return CustomChatSnowflakeCortex(
            model=model,
            cortex_function="try_complete",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            session=session,
            snowflake_username="oauth_user",
        )