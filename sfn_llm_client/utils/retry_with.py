import time
from functools import wraps
from typing import Callable, Any
from sfn_llm_client.utils.logging import setup_logger

def retry_with(
    retries: int = 3,
    retry_delay: float = 3.0,
    backoff: bool = False
) -> Callable:
    """
    A Retry decorator
    Args:
        retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        backoff: If True, applies exponential backoff to retry delay.
        logger: Logger instance for logging retries and errors.
    """
    logger, _ = setup_logger(logger_name="RetryDecorator")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = retry_delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed: {str(e)}. Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        if backoff:
                            delay *= 2  # Exponential backoff
                    else:
                        logger.error(
                            f"Attempt {attempt + 1}/{retries} failed: {str(e)}. No more retries left."
                        )
                        raise
        return wrapper
    return decorator
