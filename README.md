# SFN_LLM_Client

This is an enhanced and improved version with latest llm provider chat completion feature The `sfn_llm_client` now includes:

- **Updated to the latest version of OpenAI**.
- **Integrated Cortex LLM provider support**.
- **Latest improvements and updates to the codebase** for better performance and compatibility.

## Features

- Supports multiple LLM providers, including OpenAI and Cortex.
- Easily extensible to include new LLM providers by implementing base client classes.
- Well-documented and tested.

### Adding a New LLM Client

To add a new LLM client, follow these steps:

1. **Implement `BaseLLMClient` or `BaseLLMAPIClient`:**  
   If you're adding a new LLM provider, you'll need to implement either the `BaseLLMClient` or `BaseLLMAPIClient` interfaces.
   
2. **Register in `LLMAPIClientFactory`:**  
   If you're adding a client based on `BaseLLMAPIClient`, don't forget to register it in the `LLMAPIClientFactory` so that it's available for use.

### Adding Dependencies

If your LLM client requires additional dependencies, you can add them to the `pyproject.toml` file under the appropriate section.

## Contributing
Contributions are welcome! If you'd like to help improve this SDK, please check out the todos or open an issue or pull request.

### Credits
the core forked functionality taken from `llm-client-sdk` created by uripeled2.

## Contact:
For any queries or issues, please contact the maintainer at: `rajesh@stepfunction.ai`