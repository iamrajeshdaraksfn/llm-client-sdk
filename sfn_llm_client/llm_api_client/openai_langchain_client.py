from typing import Optional
from langchain_openai import ChatOpenAI
from sfn_llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage
from sfn_llm_client.utils.consts import PROMPT_KEY
from sfn_llm_client.llm_cost_calculation.openai_cost_calculation import openai_cost_calculation
from sfn_llm_client.utils.logging import setup_logger
from sfn_llm_client.utils.retry_with import retry_with

class OpenAILangchainClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        self.logger, _ = setup_logger(logger_name="OpenAILangchainClient")
        self._client = None # Stores the single ChatOpenAI client instance
        self._client_params = None # Stores the parameters of the current client
    
    def get_or_create_client(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """
        Gets the existing ChatOpenAI client or creates a new one if it doesn't exist
        or if the requested parameters are different from the current client's parameters.

        Args:
            api_key (str): The OpenAI API key.
            model (str): The OpenAI model name (e.g., "gpt-3.5-turbo").
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI client.
        """
        requested_params = (api_key, model, temperature, max_tokens)

        # Check if a client already exists and if its parameters match the requested ones.
        if self._client and self._client_params == requested_params:
            print(f"Reusing existing client with parameters: {requested_params}")
            return self._client
        else:
            # If no client exists, or if parameters are different, create a new one.
            print(f"Creating new client for parameters: {requested_params}")
            try:
                # temperature not supported in following models
                if model in ("o1-mini", "o1-pro", "o3", "o4-mini", "o3-mini"):
                    client = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens
                    )
                else:
                    client = ChatOpenAI(
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                self._client = client
                self._client_params = requested_params
                return self._client
            except Exception as e:
                print(f"Error creating ChatOpenAI client: {e}")
                raise
    # TODO: try if get_or_create_client can be used instead of get_langchain_llm
    def get_langchain_llm(self, api_key=None, model=None, temperature=None, max_tokens=None):
        """
        Returns the underlying ChatOpenAI client, creating it if necessary.
        """
        api_key = api_key or self._api_key
        model = model or self._default_model
        temperature = temperature if temperature is not None else 0
        max_tokens = max_tokens if max_tokens is not None else 16
        return self.get_or_create_client(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)

    @retry_with(retries=3, retry_delay=3.0, backoff=True)
    def chat_completion(self, messages: list[ChatMessage], temperature: float = 0,
                        max_tokens: int = 16, top_p: float = 1, model: Optional[str] = None, 
                        retries: int = 3, retry_delay: float = 3.0, **kwargs) -> list[str]:
        """
        This method performs chat completion with OpenAI, and includes basic retry logic for handling
        exceptions or empty responses.

        :param retries: Number of retries in case of failure.
        :param retry_delay: Delay in seconds between retries.
        """
        self._set_model_in_kwargs(kwargs, model)
        messages = [
            message if isinstance(message, dict) else message.to_dict() 
            for message in messages
        ]
        self.get_or_create_client(api_key=self._api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        
        completions = self._client.invoke(messages)
        print("completions:" , completions)

        # Check if response is empty
        if not completions or not completions.content:
            raise ValueError("Received empty response from the openai llm")

        token_usage = completions.response_metadata["token_usage"]
        prompt_tokens = token_usage['prompt_tokens']
        completion_tokens = token_usage['completion_tokens']
        
        token_cost_summary = openai_cost_calculation(
            prompt_tokens,
            completion_tokens,
            model,
        )
        return completions, token_cost_summary
    
    async def text_completion(self, *args, **kwargs):
        raise NotImplementedError("text_completion is not supported in OpenAILangchainClient.")

    async def embedding(self, *args, **kwargs):
        raise NotImplementedError("embedding is not supported in OpenAILangchainClient.")

    async def get_chat_tokens_count(self, *args, **kwargs):
        raise NotImplementedError("get_chat_tokens_count is not supported in OpenAILangchainClient.")