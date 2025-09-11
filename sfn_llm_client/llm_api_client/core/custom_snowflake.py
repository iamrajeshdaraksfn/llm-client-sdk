import json
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
from langchain_core.runnables import Runnable
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import Field, SecretStr, model_validator
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain_community.chat_models.snowflake import (
    ChatSnowflakeCortexError,
    _convert_message_to_dict,
    _truncate_at_stop_tokens,
)





class CustomChatSnowflakeCortex(ChatSnowflakeCortex):
    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Overrides the parent validator to allow for dynamic session management.

        If a 'session' object is already provided in the values (e.g., during
        initialization), it bypasses the parent's
        logic of creating a local session from environment variables in _generate, and _stream.
        """
        # If a session is already present, we don't take the session we overide the session we dont care session any more.


        if "session" in values and values["session"] is not None:
            return values

        print("INFO: Custom validator is bypassing session creation at initialization.")
        return values
    
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        session = None
        if run_manager and run_manager.metadata:
            session = run_manager.metadata.get("custom_session")

        if not session:
            raise ValueError(
                "A Snowpark session was not found in the config metadata. "
                "Please provide it when calling the model, like so: \n"
                "llm.invoke(..., config={'metadata': {'session': your_session}})"
            )
        
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Check for tool invocation in the messages and prepare for tool use
        tool_output = None
        for message in messages:
            if (
                isinstance(message.content, dict)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content.get("invoke_tool"))
                tool_name = tool_info.get("tool_name")
                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_output = self.test_tools[tool_name](**tool_args)
                    break

        # Prepare messages for SQL query
        if tool_output:
            message_dicts.append(
                {"tool_output": str(tool_output)}
            )  # Ensure tool_output is a string

        # JSON dump the message_dicts and options without additional escaping
        message_json = json.dumps(message_dicts)
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }
        options_json = json.dumps(options)  # JSON string of options

        # Form the SQL statement using JSON literals
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json($${message_json}$$),
                parse_json($${options_json}$$)
            ) as llm_response;
        """

        try:
            # Use the Snowflake Cortex Complete function
            session.sql(
                f"USE WAREHOUSE {session.get_current_warehouse()};"
            ).collect()
            l_rows = session.sql(sql_stmt).collect()
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex: {e}"
            )
        
        response = json.loads(l_rows[0]["LLM_RESPONSE"])
        ai_message_content = response["choices"][0]["messages"]

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response["usage"],
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        session = None
        if run_manager and run_manager.metadata:
            session = run_manager.metadata.get("custom_session")

        if not session:
            raise ValueError(
                "A Snowpark session was not found in the config metadata. "
                "Please provide it when calling the model, like so: \n"
                "llm.invoke(..., config={'metadata': {'session': your_session}})"
            )
        
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Check for and potentially use a tool before streaming
        for message in messages:
            if (
                isinstance(message, str)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content)
                tool_list = tool_info.get("invoke_tools", [])
                for tool in tool_list:
                    tool_name = tool.get("tool_name")
                    tool_args = tool.get("args", {})

                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_result = self.test_tools[tool_name](**tool_args)
                    additional_context = {"tool_output": tool_result}
                    message_dicts.append(
                        additional_context
                    )  # Append tool result to message dicts

        # JSON dump the message_dicts and options without additional escaping
        message_json = json.dumps(message_dicts)
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            # "stream": True,
        }
        options_json = json.dumps(options)  # JSON string of options

        # Form the SQL statement using JSON literals
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json($${message_json}$$),
                parse_json($${options_json}$$)
            ) as llm_stream_response;
        """

        try:
            # Use the Snowflake Cortex Complete function
            session.sql(
                f"USE WAREHOUSE {session.get_current_warehouse()};"
            ).collect()
            result = session.sql(sql_stmt).collect()

            # Iterate over the generator to yield streaming responses
            for row in result:
                response = json.loads(row["LLM_STREAM_RESPONSE"])
                ai_message_content = response["choices"][0]["messages"]

                # Stream response content in chunks
                for chunk in self._stream_content(ai_message_content, stop):
                    yield chunk

        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex stream: {e}"
            )
    
