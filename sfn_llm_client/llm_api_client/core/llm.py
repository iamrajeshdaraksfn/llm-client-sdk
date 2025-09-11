
from importlib import util
from langchain_community.chat_models import FakeListChatModel
from langchain_core.language_models import BaseChatModel
from pydantic import model_validator
from typing import Dict, Optional
from functools import cache
from langchain_core.utils.input import get_colored_text, get_bolded_text
from langchain_core.tracers.stdout import try_json_stringify


from sfn_llm_client.llm_api_client.core.model_schema import LLMConfig, Provider
from sfn_llm_client.utils.logging import setup_logger


# class FakeToolModel(FakeListChatModel):
#     def __init__(self, responses: list[str]):
#         super().__init__(responses=responses)
#     def bind_tools(self, tools):
#         return self


logger, _ = setup_logger(logger_name="LLMClient")

def _check_pkg(pkg: str, *, pkg_kebab: Optional[str] = None) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab if pkg_kebab is not None else pkg.replace("_", "-")
        msg = (
            f"Unable to import {pkg}. Please install with `pip install -U {pkg_kebab}`"
        )
        logger.error(msg)
        raise ImportError(msg)

def load_model( config: LLMConfig) -> BaseChatModel:

    provider, model = config.model_name.split("/", maxsplit=1)
    temperature=config.temperature
    top_p=config.top_p
    max_retries=config.max_retries
    timeout=config.api_timeout
    cortex_function=config.cortex_function

    if provider == Provider.OPENAI.value:
        _check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI
        logger.info("Creating OpenAI model instance: '{}'".format(model))
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            timeout=timeout
        )
    if provider == Provider.SNOWFLAKE.value:
        _check_pkg("langchain_community")
        _check_pkg("snowflake.snowpark", pkg_kebab="snowflake-snowpark-python")
        from sfn_llm_client.llm_api_client.core.custom_snowflake import CustomChatSnowflakeCortex
        logger.info("Creating Snowflake model instance: '{}'".format(model))
        return CustomChatSnowflakeCortex(
            model=model, 
            temperature=temperature,
            top_p=top_p,
            cortex_function=cortex_function,
        )
    
    msg = (
        f"Unsupported {provider}"
    )
    logger.error(msg)
    raise ValueError(msg)

    


# def _create_single_model(config: LLMConfig) -> BaseChatModel:

#     provider, model_id = config.model_name.split("/", maxsplit=1)

#     if provider == Provider.OPENAI.value:
#         if ChatOpenAI is None:
#             config.logger.error("OpenAI dependencies not found. Run 'pip install langchain-openai'")
#             raise ImportError("OpenAI dependencies not found. Run 'pip install langchain-openai'.")
        
#         config.logger.info(f"Creating OpenAI model instance: '{model_id}'")
#         return ChatOpenAI(
#             model=model_id,
#             temperature=config.temperature,
#             top_p=config.top_p,
#             max_retries=config.max_retries,
#             timeout=config.api_timeout
#         )

#     elif provider == Provider.SNOWFLAKE.value:
#         if CustomChatSnowflakeCortex is None:
#             config.logger.error("Snowflake dependencies not found. Run 'pip install langchain-community snowflake-snowpark-python'.")
#             raise ImportError("Snowflake dependencies not found. Run 'pip install langchain-community snowflake-snowpark-python'.")
        
#         config.logger.info(f"Creating Snowflake Cortex model instance: '{model_id}'")
#         return CustomChatSnowflakeCortex(
#                 model=model_id, 
#                 temperature=config.temperature,
#                 top_p=config.top_p,
#                 cortex_function=config.cortex_function,
#             )

#     elif provider == Provider.FAKE.value:
#         config.logger.info(f"Creating Fake model instance: '{model_id}'")
#         responses = config.fake_responses or ["This is a default fake response."]
#         return FakeToolModel(responses=responses)

#     config.logger.error(f"Unsupported provider: '{provider}'. This should have been caught by validation.")
#     raise ValueError(f"Unsupported provider: {provider}")




# def get_model( config: LLMConfig) -> BaseChatModel:
#     if not config:
#         temp_logger = logging.getLogger(__name__)
#         temp_logger.error("Model configuration is required but was not provided.")
#         raise ValueError("Model configuration is required")

#     primary_llm = _create_single_model(config)
#     if not config.fall_back_model:
#         config.logger.info(f"Model '{config.model_name}' loaded without a fallback.")
#         return primary_llm

#     config.logger.info(f"Preparing fallback model '{config.fall_back_model}' for primary model '{config.model_name}'.")

#     fallback_config = LLMConfig(
#         model_name=config.fall_back_model,
#         fall_back_model=config.fall_back_model, 
#         temperature=config.temperature,
#         top_p=config.top_p,
#         max_retries=2,
#         api_timeout=config.api_timeout,
#         logger=config.logger,
#     )
#     fallback_llm = _create_single_model(fallback_config)
#     return primary_llm.with_fallbacks([fallback_llm])






    
    