import json
from unittest.mock import patch
from llm_client.llm_api_client.cortex_client import snowflake_cortex_cost_calculation

def test_snowflake_cortex_cost_calculation(self):
    """Test the token cost calculation for Cortex."""
    mock_response = json.dumps({
        "choices": [{"text": "Sample response"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "guardrails_tokens": 2,
        }
    })
    model = "snowflake-arctic"
    mock_cost = {
        "TOKENS_COST": 0.02,  # Example cost per million tokens
    }

    with patch("cortex_client.CORTEX_MODEL_TOKENS_COST", {model: mock_cost}):
        token_cost_summary = snowflake_cortex_cost_calculation(
            response=mock_response, model=model
        )
        self.assertEqual(token_cost_summary["total_tokens"], 17)
        self.assertEqual(
            token_cost_summary["total_cost_in_credits"], 17 * 0.02 / 1_000_000
        )

def test_snowflake_cortex_cost_calculation_invalid_response(self):
    """Test token cost calculation with invalid response."""
    mock_response = json.dumps({})
    model = "snowflake-arctic"

    with self.assertRaises(ValueError):
        snowflake_cortex_cost_calculation(response=mock_response, model=model)

def test_snowflake_cortex_cost_calculation_unsupported_model(self):
    """Test token cost calculation with unsupported model."""
    mock_response = json.dumps({
        "choices": [{"text": "Sample response"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
    })
    model = "unsupported-model"

    with self.assertRaises(ValueError):
        snowflake_cortex_cost_calculation(response=mock_response, model=model)
