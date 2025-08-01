import logging
from langchain_community.chat_models import FakeListChatModel
from langchain_core.language_models import BaseChatModel
from pydantic import model_validator
from typing import Dict

from sfn_llm_client.llm_api_client.sfn_langgraph.model_schema import ModelConfiguration, Provider

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatSnowflakeCortex
    class CustomChatSnowflakeCortex(ChatSnowflakeCortex):
        @model_validator(mode="before")
        def validate_environment(cls, values: Dict) -> Dict:
            print("Executing custom override of validate_environment.")
            session = values.get("session")
            if session:
                print(f"DEBUG: Existing Snowpark session of type {type(session)} found. Using it directly.")
                return values

            print("DEBUG: No pre-configured session found. Deferring to parent validation logic.")
            return values

except ImportError:
    ChatSnowflakeCortex = None
    CustomChatSnowflakeCortex = None 

class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)
    def bind_tools(self, tools):
        return self


class ModelLoader:
    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance

    def load(self, config: ModelConfiguration) -> BaseChatModel:
        if not config.model_configuration:
            self.logger.error("Model configuration is required but was not provided.")
            raise ValueError("Model configuration is required")

        model_config = config.model_configuration
        provider, model_id = model_config.model_name.split("/", maxsplit=1)

        if provider == Provider.OPENAI.value:
            if ChatOpenAI is None:
                raise ImportError("OpenAI dependencies not found. Run 'pip install langchain-openai'.")
            
            self.logger.info(f"Loading OpenAI model: name '{model_id}'")
            return ChatOpenAI(
                    model=model_id,
                    temperature=model_config.temperature,
                    top_p=model_config.top_p,
                )

        elif provider == Provider.SNOWFLAKE.value:
            if CustomChatSnowflakeCortex is None:
                raise ImportError("Snowflake dependencies not found. Run 'pip install langchain-community snowflake-snowpark-python'.")
            
            self.logger.info(f"Loading Snowflake Cortex model with custom validator: '{model_id}'")
            return CustomChatSnowflakeCortex(
                    model=model_id, 
                    temperature=model_config.temperature,
                    top_p=model_config.top_p,
                    cortex_function=model_config.cortex_function,
                    session=model_config.session,
                    snowflake_username="oauth_user"
                )

        elif provider == Provider.FAKE.value:
            self.logger.info(f"Loading Fake model: '{model_id}'")
            responses = model_config.fake_responses or ["This is a default fake response."]
            return FakeToolModel(responses=responses)

        self.logger.error(f"Unsupported provider: '{provider}'. This should have been caught by validation.")
        raise ValueError(f"Unsupported provider: {provider}")