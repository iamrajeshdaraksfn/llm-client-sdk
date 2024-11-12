import os
from unittest.mock import MagicMock, patch
import pytest
from llm_client.llm_api_client.openai_client import ChatMessage
from llm_client import OpenAIClient
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig, Role

@pytest.fixture
def open_ai_client():
    """Fixture to create an OpenAIClient instance with a mock logger."""
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    model = "gpt-4o-mini"
    # Initialize the OpenAIClient with the mock configuration
    client = OpenAIClient(
        LLMAPIClientConfig(
            api_key=OPENAI_API_KEY,
            default_model=model,
            headers={}  # Use actual headers if required or mock them
        )
    )
    # Mock the logger of the OpenAIClient instance
    client.logger = MagicMock()
    # Return the client instance for use in tests
    return client

def test_chat_completion__sanity(open_ai_client):
    # Mock successful completion response
    mock_response = MagicMock()
    mock_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', return_value=mock_response):
        actual, token_consumption_dict = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")]
        )

    assert actual == mock_response
    open_ai_client.logger.error.assert_not_called()

def test_chat_completion__retry_success_after_failure(open_ai_client):
    # Mock failure for first two attempts, success on the third
    mock_response = MagicMock()
    mock_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', side_effect=[
        Exception("Temporary failure"),
        Exception("Temporary failure"),
        mock_response
    ]):
        actual, token_consumption_dict = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")],
            retries=3,
            retry_delay=1
        )

    assert actual == mock_response
    assert open_ai_client.logger.error.call_count == 2  # Two errors for retries

def test_chat_completion__max_retries(open_ai_client):
    # Mock failure for all retry attempts
    with patch('openai.Completion.create', side_effect=Exception("API failure")):
        with pytest.raises(Exception, match="API failure"):
            open_ai_client.chat_completion(
                [ChatMessage(Role.USER, "Hello!")],
                retries=3,
                retry_delay=1
            )

    assert open_ai_client.logger.error.call_count == 3  # Three retries before failure

def test_chat_completion__empty_response(open_ai_client):
    # Mock empty response first, valid response second
    mock_empty_response = MagicMock()
    mock_empty_response.choices = []

    mock_valid_response = MagicMock()
    mock_valid_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_valid_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', side_effect=[mock_empty_response, mock_valid_response]):
        actual, token_consumption_dict = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")],
            retries=2,
            retry_delay=1
        )

    assert actual == mock_valid_response
    assert open_ai_client.logger.error.call_count == 1  # One error for empty response

def test_chat_completion__no_retry_on_success(open_ai_client):
    # Mock successful response on the first attempt
    mock_response = MagicMock()
    mock_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', return_value=mock_response):
        actual, token_consumption_dict = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")],
            retries=3,
            retry_delay=1
        )

    assert actual == mock_response
    open_ai_client.logger.error.assert_not_called()  # No error, no retries

def test_chat_completion__multiple_completions(open_ai_client):
    # Mock response with multiple completions
    mock_response = MagicMock()
    mock_response.choices = [
        {"message": {"content": "Hello there, how may I assist you today?"}},
        {"message": {"content": "Second completion"}}
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', return_value=mock_response):
        actual, token_consumption_dict = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")]
        )

    assert actual == mock_response
    assert len(mock_response.choices) == 2
