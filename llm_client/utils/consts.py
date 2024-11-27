MODEL_KEY = "model"
PROMPT_KEY = "prompt"
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo-0125"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

CORTEX_MODEL_TOKENS_COST = {
    "gemma-7b": {"TOKENS_COST": 0.12},
    "jamba-instruct": {"TOKENS_COST": 0.83},
    "jamba-1.5-large": {"TOKENS_COST": 1.40},
    "jamba-1.5-mini": {"TOKENS_COST": 0.10},
    "llama3.1-405b": {"TOKENS_COST": 3.00},
    "llama3.1-70b": {"TOKENS_COST": 1.21},
    "llama3.1-8b": {"TOKENS_COST": 0.19},
    "llama3.2-1b": {"TOKENS_COST": 0.04},
    "llama3.2-3b": {"TOKENS_COST": 0.06},
    "mistral-large2": {"TOKENS_COST": 1.95},
    "mistral-7b": {"TOKENS_COST": 0.12},
    "mixtral-8x7b": {"TOKENS_COST": 0.22},
    "reka-core": {"TOKENS_COST": 5.50},
    "reka-flash": {"TOKENS_COST": 0.45},
    "snowflake-arctic": {"TOKENS_COST": 0.84}
}

# pricing for 1k tokens
OPENAI_MODEL_TOKENS_COST = {
        "gpt-3.5-turbo-0125": {
            "prompt": 0.0005,
            "completion": 0.0015,
        },
        "gpt-3.5-turbo-16k": {
            "prompt": 0.003,
            "completion": 0.004,
        },
        "gpt-3.5-turbo": {
            "prompt": 0.003,
            "completion": 0.006,
        },
        "gpt-4-8k": {
            "prompt": 0.03,
            "completion": 0.06,
        },
        "gpt-4-32k": {
            "prompt": 0.06,
            "completion": 0.12,
        },
        "gpt-4o": {
            "prompt": 0.005,
            "completion": 0.015,
        },
        "gpt-4o-mini": {
            "prompt": 0.00015,
            "completion": 0.0006,
        },
        "text-embedding-ada-002-v2": {
            "prompt": 0.0001,
            "completion": 0.0001,
        },
    }

# cost token per million
ANTHROPIC_MODEL_TOKENS_COST = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
}