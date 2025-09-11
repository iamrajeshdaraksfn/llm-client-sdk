import logging
from strenum import StrEnum
from typing import Optional, Any, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    SNOWFLAKE = "snowflake"
    FAKE = "fake"

class OpenAIModelName(StrEnum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    O3_MINI = "o3-mini"
    O3 = "o3"
    O1_MINI = "o1-mini"
    O1 = "o1"
    GPT_4_1_NANO = "gpt-4-1-nano"

class SnowflakeModelName(StrEnum):
    SNOWFLAKE_ARCTIC = "snowflake-arctic"

class FakeModelName(StrEnum):
    FAKE = "fake"

MODEL_COST_PER_1M_TOKENS = {
    Provider.OPENAI.value: {
        "gpt-4o-mini": {"input_cost": 0.15, "output_cost": 0.60},
        "gpt-4o": {"input_cost": 5.00, "output_cost": 15.00},
        "gpt-4-turbo": {"input_cost": 10.00, "output_cost": 30.00},
        "gpt-4": {"input_cost": 30.00, "output_cost": 60.00},
        "gpt-4-1-nano": {"input_cost": 0.10, "output_cost": 0.40},
        "o3-mini": {"input_cost": 1.00, "output_cost": 4.00},
        "o3": {"input_cost": 2.00, "output_cost": 8.00},     
        "o1-mini": {"input_cost": 3.00, "output_cost": 12.00},
        "o1": {"input_cost": 15.00, "output_cost": 60.00},
    },
    Provider.SNOWFLAKE.value: {
        "snowflake-arctic": {"cost": 0.84},
    },
}

PROVIDER_TO_BASE_CLASS = {
    Provider.OPENAI: "ChatOpenAI",
    Provider.SNOWFLAKE: "CustomChatSnowflakeCortex",
    Provider.FAKE: "FakeToolModel",
}

SUPPORTED_MODELS_BY_PROVIDER = {
    Provider.OPENAI: OpenAIModelName,
    Provider.SNOWFLAKE: SnowflakeModelName,
    Provider.FAKE: FakeModelName,
}



class LLMConfig(BaseModel):
    model_name: str = Field(
        ...,
        description='Name of model in "provider/model" format (e.g., "openai/gpt-4o-mini").'
    )
    # fall_back_model: Optional[str] = Field(
    #     default=None,
    #     description='Optional fallback model in "provider/model" format to use if the main model fails.',
    # )

    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Controls randomness in generation."
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter."
    )
    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retries for API calls."
    )
    api_timeout: Optional[int] = Field(
        default=None, ge=0, description="Timeout for API calls in seconds."
    )

    cortex_function: str = Field(
        default="complete",
        description="The Snowflake Cortex function to use for generation.",
        json_schema_extra={"x_oap_ui_config": {"type": "string", "description": "Snowflake Cortex function"}},
    )


    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("model_name")
    @classmethod
    def _validate_model_string_format(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None

        parts = v.split("/", 1)
        if len(parts) != 2:
            raise ValueError('must be in "provider/model_name" format')

        provider_str, model_id = parts
        if provider_str not in SUPPORTED_MODELS_BY_PROVIDER:
            supported = list(SUPPORTED_MODELS_BY_PROVIDER.keys())
            raise ValueError(f'Invalid provider "{provider_str}". Supported: {supported}')

        model_enum = SUPPORTED_MODELS_BY_PROVIDER[provider_str]
        supported_models = [m.value for m in model_enum]
        if model_id not in supported_models:
            raise ValueError(
                f'Unsupported model "{model_id}" for provider "{provider_str}". '
                f"Supported: {supported_models}"
            )
        return v

