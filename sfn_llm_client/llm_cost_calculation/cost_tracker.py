import logging
import threading
from uuid import UUID
from typing import Any, Dict, List, Optional

from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler

from sfn_llm_client.llm_api_client.core.model_schema import MODEL_COST_PER_1M_TOKENS, PROVIDER_TO_BASE_CLASS, Provider

class CostCallbackHandler(BaseCallbackHandler):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guardrails_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(
        self,
        logger: logging.Logger,
        cost_dict: Dict = MODEL_COST_PER_1M_TOKENS,
        provider_map: Dict = PROVIDER_TO_BASE_CLASS,
    ) -> None:
        super().__init__()
        self.logger = logger
        self._lock = threading.Lock()
        self.cost_dict = cost_dict
        self._base_class_to_provider = {v: k for k, v in provider_map.items()}
        self._run_info: Dict[UUID, Dict[str, Any]] = {}

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"\tGuardrails Tokens: {self.guardrails_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost:.6f}"
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        class_hierarchy = serialized.get("id", [])
        model_provider = next(
            (self._base_class_to_provider[cls] for cls in class_hierarchy if cls in self._base_class_to_provider),
            None,
        )
        model_id = serialized.get("kwargs", {}).get("model_name")

        if model_provider == Provider.SNOWFLAKE:
            model_id = kwargs.get("metadata", {}).get("ls_model_name", None)            

        if not model_provider or not model_id:
            self.logger.warning(
                f"Could not determine provider or model for run {run_id}. "
                f"Cost will not be calculated. Class Hierarchy: {class_hierarchy}, Model ID: {model_id}"
            )
            return

        if model_provider == Provider.FAKE:
            self.logger.debug(f"Skipping cost calculation for Fake model run {run_id}.")
            return

        with self._lock:
            self._run_info[run_id] = {"provider": model_provider, "model_name": model_id}

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            run_info = self._run_info.pop(run_id, None)

        if not run_info:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"No start information for run {run_id}. This is normal for fake models.")
            return
        
        provider: Provider = run_info["provider"]
        model_name: str = run_info["model_name"]
        
        if provider == Provider.SNOWFLAKE:
            generations = getattr(response, 'generations', [])
            generation = (generations[0] if generations else [{}])[0]
            message = getattr(generation, 'message', None)
            token_usage = getattr(message, 'response_metadata', {}) 

        else:
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage")

        if not token_usage:
            self.logger.warning(f"No token_usage information in LLM output for run {run_id}. Cost cannot be calculated.")
            return

        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        guardrails_tokens = token_usage.get("guardrails_tokens", 0) if provider == Provider.SNOWFLAKE else 0
        
        run_cost = 0.0
        model_costs = self.cost_dict.get(provider.value, {}).get(model_name)

        if model_costs:
            if provider == Provider.SNOWFLAKE:
                total_run_tokens = prompt_tokens + completion_tokens + guardrails_tokens
                cost_per_mil = model_costs.get("cost", 0.0)
                run_cost = (total_run_tokens / 1_000_000) * cost_per_mil
            else:
                input_cost_per_mil = model_costs.get("input_cost", 0.0)
                output_cost_per_mil = model_costs.get("output_cost", 0.0)
                run_cost = (prompt_tokens * input_cost_per_mil + completion_tokens * output_cost_per_mil) / 1_000_000
        else:
             self.logger.warning(
                f"Cost not found for model '{model_name}' from provider '{provider.value}'. "
                f"Cost will be zero for this run."
            )

        with self._lock:
            self.total_tokens += prompt_tokens + completion_tokens + guardrails_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.guardrails_tokens += guardrails_tokens
            self.total_cost += run_cost
            self.successful_requests += 1
    
    def reset(self) -> None:
        with self._lock:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.guardrails_tokens = 0
            self.successful_requests = 0
            self.total_cost = 0.0
            self.logger.info("Cost tracker has been reset.")

    def __copy__(self) -> "CostCallbackHandler":
        return self

    def __deepcopy__(self, memo: Any) -> "CostCallbackHandler":
        return self