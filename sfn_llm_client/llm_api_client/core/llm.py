from langchain_community.chat_models import FakeListChatModel
from langchain_core.language_models import BaseChatModel
from pydantic import model_validator
from typing import Dict
from functools import cache

from sfn_llm_client.llm_api_client.core.model_schema import LLMConfig, Provider

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



def _create_single_model(config: LLMConfig) -> BaseChatModel:

    provider, model_id = config.model_name.split("/", maxsplit=1)

    if provider == Provider.OPENAI.value:
        if ChatOpenAI is None:
            config.logger.error("OpenAI dependencies not found. Run 'pip install langchain-openai'")
            raise ImportError("OpenAI dependencies not found. Run 'pip install langchain-openai'.")
        
        config.logger.info(f"Creating OpenAI model instance: '{model_id}'")
        return ChatOpenAI(
            model=model_id,
            temperature=config.temperature,
            top_p=config.top_p,
            max_retries=config.max_retries,
            timeout=config.api_timeout
        )

    elif provider == Provider.SNOWFLAKE.value:
        if CustomChatSnowflakeCortex is None:
            config.logger.error("Snowflake dependencies not found. Run 'pip install langchain-community snowflake-snowpark-python'.")
            raise ImportError("Snowflake dependencies not found. Run 'pip install langchain-community snowflake-snowpark-python'.")
        
        config.logger.info(f"Creating Snowflake Cortex model instance: '{model_id}'")
        return CustomChatSnowflakeCortex(
                model=model_id, 
                temperature=config.temperature,
                top_p=config.top_p,
                cortex_function=config.cortex_function,
                session=config.session,
                snowflake_username="oauth_user"
            )

    elif provider == Provider.FAKE.value:
        config.logger.info(f"Creating Fake model instance: '{model_id}'")
        responses = config.fake_responses or ["This is a default fake response."]
        return FakeToolModel(responses=responses)

    config.logger.error(f"Unsupported provider: '{provider}'. This should have been caught by validation.")
    raise ValueError(f"Unsupported provider: {provider}")




def get_model( config: LLMConfig) -> BaseChatModel:
    if not config:
        temp_logger = logging.getLogger(__name__)
        temp_logger.error("Model configuration is required but was not provided.")
        raise ValueError("Model configuration is required")

    primary_llm = _create_single_model(config)
    if not config.fall_back_model:
        config.logger.info(f"Model '{config.model_name}' loaded without a fallback.")
        return primary_llm

    config.logger.info(f"Preparing fallback model '{config.fall_back_model}' for primary model '{config.model_name}'.")

    fallback_config = LLMConfig(
        model_name=config.fall_back_model,
        fall_back_model=config.fall_back_model, 
        temperature=None,
        top_p=None,
        max_retries=2,
        api_timeout=None,
        logger=config.logger,
        session=config.session 
    )
    fallback_llm = _create_single_model(fallback_config)
    return primary_llm.with_fallbacks([fallback_llm])