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
    response = json.loads(response)
    # Check if response is empty
    if not response or not response['choices']:
        raise ValueError("Received empty response from the openai llm")

    # Ensure model is supported
    if model not in CORTEX_MODEL_TOKENS_COST:
        raise ValueError(f"Unsupported model: {model}")

    # Extract token usage from response
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    guardrails_tokens = response['usage'].get('guardrails_tokens', 0)  # Handle missing guardrails_tokens
    
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