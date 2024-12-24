from sfn_llm_client.utils.consts import ANTHROPIC_DEFAULT_MODEL, ANTHROPIC_MODEL_TOKENS_COST
from sfn_llm_client.utils.logging import setup_logger

# Initialize logger
logger, _ = setup_logger(logger_name="AnthropicCostCalculation")

def anthropic_cost_calculation(
    total_input_tokens: int,
    total_output_tokens: int,
    model: str = ANTHROPIC_DEFAULT_MODEL
) -> dict:
    """
    This method calculates usage cost of Anthropic model, based on the number of tokens used.

    Args:
        total_input_tokens: The total number of tokens in the input.
        total_output_tokens: The total number of tokens in the model's response.
        model: The name of the Anthropic model to use or Defaults to ANTHROPIC_DEFAULT_MODEL.

    Returns:
        A dictionary containing the following keys:
            - prompt_tokens: The total number of input tokens.
            - completions_tokens: The total number of output tokens.
            - total_tokens: The total number of tokens.
            - total_cost_usd: The total cost in USD.
    """
    logger.info(f'started Anthropic cost calculation with model: {model}')

    try:
        model_pricing = ANTHROPIC_MODEL_TOKENS_COST[model]
    except KeyError:
        raise ValueError(f"Invalid model specified: {model}")

    # Calculate costs
    input_cost = total_input_tokens * model_pricing["input"] / 1000
    output_cost = total_output_tokens * model_pricing["output"] / 1000
    total_tokens = total_input_tokens + total_output_tokens
    total_cost_usd = input_cost + output_cost

    token_cost_summary = {
        "prompt_tokens": total_input_tokens,
        "completion_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost_usd, 4),
    }
    logger.info(f'Anthropic cost calculation done! total cost is: {token_cost_summary}')
    return token_cost_summary