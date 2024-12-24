import json
from unittest.mock import patch, MagicMock
from sfn_llm_client.llm_api_client.base_llm_api_client import ChatMessage, Role
from snowflake.snowpark import Session
from sfn_llm_client.llm_api_client.cortex_client import CortexClient

@patch("cortex_client.Complete")
@patch("cortex_client.setup_logger")
def test_chat_completion_success(self, mock_logger, mock_complete):
    """Test the chat_completion method for successful API call."""
    # Mock logger
    mock_logger.return_value = (MagicMock(), MagicMock())

    # Mock response
    mock_response = {
        "choices": [{"text": "Sample response"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "guardrails_tokens": 0,
        }
    }
    mock_complete.return_value = json.dumps(mock_response)

    # Create CortexClient instance
    client = CortexClient()
    
    # Mock session
    mock_session = MagicMock(spec=Session)
    
    # Call chat_completion
    messages = [ChatMessage(role=Role.USER, content="Hello, Cortex!")]
    completions, token_cost_summary = client.chat_completion(
        messages=messages, session=mock_session
    )
    
    # Assert the results
    self.assertEqual(completions, json.dumps(mock_response))
    self.assertEqual(token_cost_summary["total_tokens"], 15)
    self.assertIn("total_cost_in_credits", token_cost_summary)
    mock_complete.assert_called_once()

@patch("cortex_client.Complete")
@patch("cortex_client.setup_logger")
def test_chat_completion_failure(self, mock_logger, mock_complete):
    """Test the chat_completion method for API failure."""
    # Mock logger
    mock_logger.return_value = (MagicMock(), MagicMock())

    # Mock response with empty choices
    mock_response = {"choices": [], "usage": {}}
    mock_complete.return_value = json.dumps(mock_response)

    client = CortexClient()

    # Mock session
    mock_session = MagicMock(spec=Session)
    
    messages = [ChatMessage(role=Role.USER, content="Hello, Cortex!")]

    with self.assertRaises(ValueError):
        client.chat_completion(messages=messages, session=mock_session)
