from strenum import StrEnum
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

try:
    from snowflake.snowpark.session import Session
except ImportError:
    Session = Any 

class Provider(StrEnum):
    OPENAI = "openai"
    SNOWFLAKE = "snowflake"
    FAKE = "fake"

class OpenAIModelName(StrEnum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

class SnowflakeModelName(StrEnum):
    SNOWFLAKE_ARCTIC = "snowflake-arctic"

class FakeModelName(StrEnum):
    FAKE = "fake"

MODEL_COST_PER_1M_TOKENS = {
    Provider.OPENAI.value: {
        "gpt-4o-mini": {"input_cost": 0.15, "output_cost": 0.60},
        "gpt-4o": {"input_cost": 5.00, "output_cost": 15.00},
    },
    Provider.SNOWFLAKE.value: {
        "snowflake-arctic": {"cost": 0.84},
    },
    "anthropic": {
        "claude-3-opus-20240229": {"input_cost": 15.00, "output_cost": 75.00},
        "claude-3.5-sonnet-20240620": {"input_cost": 3.00, "output_cost": 15.00},
        "claude-3-haiku-20240307": {"input_cost": 0.25, "output_cost": 1.25},
    }
}



PROVIDER_TO_BASE_CLASS = {
    Provider.OPENAI: "ChatOpenAI",
    Provider.SNOWFLAKE: "ChatSnowflakeCortex",
    Provider.FAKE: "FakeToolModel",
}

SUPPORTED_MODELS_BY_PROVIDER = {
    Provider.OPENAI: OpenAIModelName,
    Provider.SNOWFLAKE: SnowflakeModelName,
    Provider.FAKE: FakeModelName,
}

class ModelConfig(BaseModel):
    model_name: str = Field(
        ...,
        description='Name of model in "provider/model" format (e.g., "openai/gpt-4o-mini").',
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "description": 'Model in "provider/model" format (required).'
            }
        }
    )

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None)

    cortex_function: str = Field(
        default="complete",
        json_schema_extra={"x_oap_ui_config": {"type": "string", "description": "Snowflake Cortex function"}}
    )
    
    fake_responses: Optional[list[str]] = Field(
        default=None,
        description="A list of canned responses for the fake model."
    )
    
    session: Optional[Session] = Field(
        default=None, 
        description="Snowpark session object required for Snowflake models.",
        exclude=True
    )

    @model_validator(mode='after')
    def validate_model_and_dependencies(self) -> 'ModelConfig':
        if "/" not in self.model_name:
            raise ValueError('model_name must be in "provider/model_name" format.')
        
        provider_str, model_id = self.model_name.split("/", maxsplit=1)
            
        if provider_str not in SUPPORTED_MODELS_BY_PROVIDER:
            supported_providers = list(SUPPORTED_MODELS_BY_PROVIDER.keys())
            raise ValueError(f"Unknown provider '{provider_str}'. Supported providers are: {supported_providers}")

        model_enum = SUPPORTED_MODELS_BY_PROVIDER[provider_str]
        supported_models = [m.value for m in model_enum]
        
        if model_id not in supported_models:
            raise ValueError(
                f"Unsupported model '{model_id}' for provider '{provider_str}'. "
                f"Supported models for this provider are: {supported_models}"
            )

        if provider_str == Provider.SNOWFLAKE.value:
            if not self.session:
                raise ValueError("A valid Snowpark 'session' object is required for the Snowflake provider.")
            
            if Session is not Any and not isinstance(self.session, Session):
                raise TypeError(
                    f"The 'session' object must be of type snowflake.snowpark.Session, "
                    f"not {type(self.session).__name__}"
                )
        
        return self

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)



class ModelConfiguration(BaseModel):
    model_configuration: Optional[ModelConfig] = Field(default=None)