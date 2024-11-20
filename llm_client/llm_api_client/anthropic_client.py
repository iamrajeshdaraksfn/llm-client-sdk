from typing import Optional
import time
from anthropic import Anthropic

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage, Role
from llm_client.utils.consts import PROMPT_KEY
from llm_client.utils.logging import setup_logger

COMPLETE_PATH = "complete"
BASE_URL = "https://api.anthropic.com/v1/"
COMPLETIONS_KEY = "completion"
AUTH_HEADER = "x-api-key"
ACCEPT_HEADER = "Accept"
VERSION_HEADER = "anthropic-version"
ACCEPT_VALUE = "application/json"
MAX_TOKENS_KEY = "max_tokens_to_sample"
USER_PREFIX = "Human:"
ASSISTANT_PREFIX = "Assistant:"
START_PREFIX = "\n\n"
SYSTEM_START_PREFIX = "<admin>"
SYSTEM_END_PREFIX = "</admin>"


class AnthropicClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        self._base_url = config.base_url or BASE_URL
        self._anthropic = Anthropic(api_key=config.api_key)
        self.logger, _ = setup_logger(logger_name="OpenAIClient")
        self._headers = {
            VERSION_HEADER: self._anthropic.default_headers[VERSION_HEADER],
            ACCEPT_HEADER: ACCEPT_VALUE,
            AUTH_HEADER: config.api_key,
        }

    def chat_completion(self, messages: list[ChatMessage], model: Optional[str] = None,
                    max_tokens: Optional[int] = None, temperature: float = 1.0, retries: int = 3,
                    retry_delay: float = 3.0, **kwargs) -> list[str]:
        """
        This method performs chat completion with retry logic for handling exceptions or empty responses.
        It also calculates token consumption based on the API response.

        :param messages: List of ChatMessage objects representing the conversation history.
        :param model: Optional model name to be used.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :param temperature: Controls the creativity of the response.
        :param retries: Number of retries in case of failure.
        :param retry_delay: Delay in seconds between retries.

        :return: ChatCompletion object containing the generated text and other metadata,
                or None if all retries fail.
        """

        self.logger.info("Started running llm client sdk chat completion...")
        self._set_model_in_kwargs(kwargs, model)

        prompt = self.messages_to_text(messages)

        attempt = 0
        while attempt < retries:
            try:
                completions = self.text_completion(
                    prompt,
                    model,
                    max_tokens,
                    temperature,
                    **kwargs
                )

                # Check if response is empty
                if not completions or not completions.choices:
                    raise ValueError("Received empty response from the API")

                # Calculate token consumption
                tokens = self.get_chat_tokens_count(messages)
                print('tokens here -------------------',tokens)
                # token_cost_summary = llm_cost_calculation(
                #     completions.usage.prompt_tokens,
                #     completions.usage.completion_tokens,
                #     model=model,
                # )

                # return completions, token_cost_summary
                return completions

            except Exception as e:
                attempt += 1
                self.logger.error(f"Error in chat_completion (attempt {attempt}/{retries}): {e}")

                if attempt >= retries:
                    self.logger.error("Max retries reached. Unable to complete request.")
                    raise e  # Reraise the exception after max retries

                time.sleep(retry_delay)  # Wait before retrying

        # If all retries fail, return None
        return None

    def text_completion(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                         temperature: float = 1.0, top_p: Optional[float] = None, **kwargs) -> list[str]:
        if max_tokens is None and kwargs.get(MAX_TOKENS_KEY) is None:
            raise ValueError(f"max_tokens or {MAX_TOKENS_KEY} must be specified")

        kwargs[PROMPT_KEY] = prompt
        kwargs[MAX_TOKENS_KEY] = kwargs.pop(MAX_TOKENS_KEY, max_tokens)
        kwargs["temperature"] = temperature
        if top_p:
            kwargs["top_p"] = top_p

        self._set_model_in_kwargs(kwargs, model)
        response = self._anthropic.messages.create(model=model, messages=[{"content": prompt}], **kwargs)
        return [response[COMPLETIONS_KEY]]

    def get_chat_tokens_count(self, messages: list[ChatMessage], **kwargs) -> int:
        return self.get_tokens_count(self.messages_to_text(messages), **kwargs)

    def get_tokens_count(self, text: str, **kwargs) -> int:
        return self._anthropic.count_tokens(text)
        # return sum(len(word.split()) for word in text.split("\n"))  # Approximate token count based on words

    def messages_to_text(self, messages: list[ChatMessage]) -> str:
        prompt = START_PREFIX
        prompt += START_PREFIX.join(map(self._message_to_prompt, messages))
        if messages[-1].role != Role.ASSISTANT:
            prompt += START_PREFIX
            prompt += self._message_to_prompt(ChatMessage(role=Role.ASSISTANT, content=""))
        return prompt.rstrip()

    @staticmethod
    def _message_to_prompt(message: ChatMessage) -> str:
        if message.role == Role.USER:
            return f"{USER_PREFIX} {message.content}"
        elif message.role == Role.ASSISTANT:
            return f"{ASSISTANT_PREFIX} {message.content}"
        elif message.role == Role.SYSTEM:
            return f"{USER_PREFIX} {SYSTEM_START_PREFIX}{message.content}{SYSTEM_END_PREFIX}"
        else:
            raise ValueError(f"Unknown role: {message.role}")