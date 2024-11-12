from llm_client.consts import OPENAI_DEFAULT_MODEL
from llm_client.logging import setup_logger

# Initialize logger
logger, _ = setup_logger(logger_name="OpenAICostCalculation")

def openai_cost_calculation(
    total_prompt_tokens: int,
    total_completion_tokens: int,
    model: str = OPENAI_DEFAULT_MODEL
) -> dict:
    """
    This method calculates usage cost of OpenAI model, based on the number of tokens used.

    Args:
        total_prompt_tokens: The total number of tokens in the prompt.
        total_completion_tokens: The total number of tokens in the model's response.
        model: The name of the OpenAI model to use or Defaults to OPENAI_DEFAULT_MODEL.

    Returns:
        A dictionary containing the following keys:
            - prompt_tokens: The total number of prompt tokens.
            - completion_tokens: The total number of completion tokens.
            - total_tokens: The total number of tokens.
            - total_Cost_usd: The total cost in USD.
    """
    logger.info(f'started openai cost calculation with model: {model}')

    # pricing for 1k tokens
    pricing = {
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

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError(f"Invalid model specified: {model}")

    # Calculate costs
    prompt_cost = total_prompt_tokens * model_pricing["prompt"] / 1000
    completion_cost = total_completion_tokens * model_pricing["completion"] / 1000
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_cost_usd = prompt_cost + completion_cost

    token_consumption_dict = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_Cost_usd": round(total_cost_usd, 4),
    }
    logger.info(f'openai cost calculation done! total cost is: {token_consumption_dict}')
    return token_consumption_dict
