
# __version__ = "0.2.0a1"  


# load utils
try:
    from .utils.base_llm_client import BaseLLMClient
    from .utils.consts import BaseLLMClient
    from .utils.base_llm_client import BaseLLMClient
    from .utils.base_llm_client import BaseLLMClient

except ImportError:
    pass

# load api clients
try:
    from .llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage, Role
    from .llm_api_client.llm_api_client_factory import LLMAPIClientFactory, LLMAPIClientType
    # load base-api clients
    try:
        from .llm_api_client.ai21_client import AI21Client
        from .llm_api_client.aleph_alpha_client import AlephAlphaClient
        from .llm_api_client.google_client import GoogleClient, MessagePrompt
    except ImportError:
        pass
    # load apis with different dependencies
    try:
        from .llm_api_client.openai_client import OpenAIClient
    except ImportError:
        pass
    try:
        from .llm_api_client.huggingface_client import HuggingFaceClient
    except ImportError:
        pass
    try:
        from .llm_api_client.anthropic_client import AnthropicClient
    except ImportError:
        pass
    try:
        from .llm_api_client.cortex_client import CortexClient
    except ImportError:
        pass
    try:
        from .llm_cost_calculation.openai_cost_calculation import openai_cost_calculation
    except ImportError:
        pass
    try:
        from .llm_cost_calculation.snowflake_cortex_cost_calculation import snowflake_cortex_cost_calculation
    except ImportError:
        pass
    try:
        from .llm_cost_calculation.anthropic_cost_calculation import anthropic_cost_calculation
    except ImportError:
        pass
except ImportError:
    pass
# load local clients
try:
    from .llm_client.local_client import LocalClient, LocalClientConfig
except ImportError:
    pass
# load sync support
try:
    from .sync.sync_llm_api_client_factory import init_sync_llm_api_client
except ImportError:
    pass



try:
    from .llm_api_client.core.llm import load_model
    from .llm_api_client.core.model_schema import  LLMConfig
    from .llm_cost_calculation.cost_tracker import CostCallbackHandler
except ImportError:
    pass