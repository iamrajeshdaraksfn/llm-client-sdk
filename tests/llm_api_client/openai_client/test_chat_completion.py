import os
from unittest.mock import MagicMock, patch
import pytest
from sfn_llm_client.llm_api_client.openai_client import ChatMessage
from sfn_llm_client import OpenAIClient
from sfn_llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig, Role

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo-0125"
    
@pytest.fixture
def open_ai_client():
    """Fixture to create an OpenAIClient instance with a mock logger."""
    client = OpenAIClient(
        LLMAPIClientConfig(
            api_key=OPENAI_API_KEY,
            default_model=MODEL,
            headers={}
        )
    )
    # Mock the logger of the OpenAIClient instance
    client.logger = MagicMock()
    # Return the client instance for use in tests
    return client

def test_chat_completion__sanity(open_ai_client):
    # Mock a successful completion response from OpenAI
    mock_response = MagicMock()
    mock_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.ChatCompletion.create', return_value=mock_response):
        actual, token_cost_summary = open_ai_client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Hello!")],
            temperature=0.7,
            max_tokens=20,
            model="gpt-4",
        )

    assert actual == mock_response  # Ensure the actual response matches the mock
    assert token_cost_summary is not None  # Ensure tokens were calculated
    open_ai_client.logger.error.assert_not_called()  # Ensure no errors were logged

def test_chat_completion__empty_response(open_ai_client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    response = open_ai_client.chat_completion(
        messages=messages,
        model=MODEL,
        temperature=0.7
    )
    
    assert response is not None

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
        actual, token_cost_summary = open_ai_client.chat_completion(
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

def test_chat_completion__no_retry_on_success(open_ai_client):
    # Mock successful response on the first attempt
    mock_response = MagicMock()
    mock_response.choices = [{"message": {"content": "Hello there, how may I assist you today?"}}]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch('openai.Completion.create', return_value=mock_response):
        actual, token_cost_summary = open_ai_client.chat_completion(
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
        actual, token_cost_summary = open_ai_client.chat_completion(
            [ChatMessage(Role.USER, "Hello!")]
        )

    assert actual == mock_response
    assert len(mock_response.choices) == 2
