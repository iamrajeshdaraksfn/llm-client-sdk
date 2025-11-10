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
    "snowflake-arctic": {"TOKENS_COST": 0.84},
    "claude-3-5-sonnet": {"TOKENS_COST": 2.55}
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
        "text-embedding-ada-002-v2": {
            "prompt": 0.0001,
            "completion": 0.0001,
        },
        "gpt-4.1": {
            "prompt": 0.002,
            "completion": 0.008
        },
        "gpt-4.1-mini": {
            "prompt": 0.0004,
            "completion": 0.0016
        },
        "gpt-4.1-nano": {
            "prompt": 0.0001,
            "completion": 0.0004
        },
        "gpt-4.5-preview": {
            "prompt": 0.075,
            "completion": 0.15
        },
        "gpt-4o": {
            "prompt": 0.0025,
            "completion": 0.01
        },
        "gpt-4o-audio-preview": {
            "prompt": 0.0025,
            "completion": 0.01
        },
        "gpt-4o-realtime-preview": {
            "prompt": 0.005,
            "completion": 0.02
        },
        "gpt-4o-mini": {
            "prompt": 0.00015,
            "completion": 0.0006
        },
        "gpt-4o-mini-audio-preview": {
            "prompt": 0.00015,
            "completion": 0.0006
        },
        "gpt-4o-mini-realtime-preview": {
            "prompt": 0.0006,
            "completion": 0.0024
        },
        "o1": {
            "prompt": 0.015,
            "completion": 0.06
        },
        "o1-pro": {
            "prompt": 0.15,
            "completion": 0.6
        },
        "o3-pro": {
            "prompt": 0.02,
            "completion": 0.08
        },
        "o3": {
            "prompt": 0.002,
            "completion": 0.008
        },
        "o4-mini": {
            "prompt": 0.0011,
            "completion": 0.0044
        },
        "o3-mini": {
            "prompt": 0.0011,
            "completion": 0.0044
        },
        "o1-mini": {
            "prompt": 0.0011,
            "completion": 0.0044
        },
        "codex-mini-latest": {
            "prompt": 0.0015,
            "completion": 0.006
        },
        "gpt-4o-mini-search-preview": {
            "prompt": 0.00015,
            "completion": 0.0006
        },
        "gpt-4o-search-preview": {
            "prompt": 0.0025,
            "completion": 0.01
        },
        "computer-use-preview": {
            "prompt": 0.003,
            "completion": 0.012
        },
        "gpt-5": {
            "prompt": 1.25 / 1000,       # ~$0.00125 per 1 k tokens
            "cached_prompt": 0.125 / 1000,  # ~$0.000125 per 1 k
            "completion": 10.00 / 1000,    # ~$0.01000 per 1 k tokens
            },
        "gpt-5-mini": {
            "prompt": 0.25 / 1000,        # ~$0.00025 per 1 k
            "cached_prompt": 0.025 / 1000,# ~$0.000025 per 1 k
            "completion": 2.00 / 1000,    # ~$0.00200 per 1 k
        },
        "gpt-5-nano": {
            "prompt": 0.05 / 1000,        # ~$0.00005 per 1 k
            "cached_prompt": 0.005 / 1000,# ~$0.000005 per 1 k
            "completion": 0.40 / 1000,    # ~$0.00040 per 1 k
        },
        "gpt-5-pro": {
            "prompt": 15.00 / 1000,       # ~$0.01500 per 1 k
            "completion": 120.00 / 1000,  # ~$0.12000 per 1 k
            # Note: cached input not listed
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