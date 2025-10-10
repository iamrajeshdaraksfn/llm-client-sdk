from sfn_llm_client.utils.consts import CORTEX_MODEL_TOKENS_COST
import tiktoken
import json

def calculate_tokens(text: str, model: str) -> int:
    """Calculate the number of tokens in the given text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def snowflake_cortex_cost_calculation(response: dict, model: str) -> tuple:
    """Calculate the cost for consumed tokens for cortex llm."""
    # In Cortex prompt and completions both tokens has same cost/credits
    # So keeping the sum as total of tokens to calculate dollar bill
    # Check if response is empty

    # Extract token usage from response
    prompt_tokens = response.get('prompt_tokens', 0)
    completion_tokens = response.get('completion_tokens', 0)
    guardrails_tokens = response.get('guardrails_tokens', 0)  # Handle missing guardrails_tokens
    # Ensure model is supported
    if model not in CORTEX_MODEL_TOKENS_COST:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "guardrails_tokens": guardrails_tokens,
            "total_tokens": prompt_tokens + completion_tokens + guardrails_tokens,
            "total_cost_in_credits": 0,
        }

    
    # Calculate total tokens and cost
    total_tokens = prompt_tokens + completion_tokens + guardrails_tokens
    token_cost_in_credits = total_tokens * CORTEX_MODEL_TOKENS_COST[model]["TOKENS_COST"] / 1000000  # Cost in credits

    token_cost_summary = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "guardrails_tokens": guardrails_tokens,
        "total_tokens": total_tokens,
        "total_cost_in_credits": token_cost_in_credits
    }
    
    return token_cost_summary


def snowflake_cortex_cost_calculation_langchain(response_metadata: dict, model: str) -> tuple:
    """Calculate the cost for consumed tokens for cortex llm."""
    
    # Check if response_metadata is empty
    if not response_metadata:
        raise ValueError("Received empty response_metadata from the cortex langchain llm")

    # Ensure model is supported
    if model not in CORTEX_MODEL_TOKENS_COST:
        raise ValueError(f"Unsupported model: {model}")

    # Extract token usage from response_metadata
    prompt_tokens = response_metadata['prompt_tokens']
    completion_tokens = response_metadata['completion_tokens']
    guardrails_tokens = response_metadata.get('guardrails_tokens', 0)  # Handle missing guardrails_tokens
    
    # Calculate total tokens and cost
    total_tokens = prompt_tokens + completion_tokens + guardrails_tokens
    token_cost_in_credits = total_tokens * CORTEX_MODEL_TOKENS_COST[model]["TOKENS_COST"] / 1000000  # Cost in credits

    token_cost_summary = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "guardrails_tokens": guardrails_tokens,
        "total_tokens": total_tokens,
        "total_cost_in_credits": token_cost_in_credits
    }
    
    return token_cost_summary