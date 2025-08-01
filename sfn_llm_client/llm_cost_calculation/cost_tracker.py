import threading
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from sfn_llm_client.llm_api_client.sfn_langgraph.model_schema import MODEL_COST_PER_1M_TOKENS, PROVIDER_TO_BASE_CLASS, Provider


class CostCallbackHandler(BaseCallbackHandler):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guardrails_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(
        self,
        cost_dict: Dict = MODEL_COST_PER_1M_TOKENS,
        provider_map: Dict = PROVIDER_TO_BASE_CLASS,
        logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.cost_dict = cost_dict
        self.provider_map = provider_map
        self.logger = logger or logging.getLogger(__name__)
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
        model_provider: Optional[Provider] = None
        class_hierarchy = serialized.get("id", [])
        
        for provider, base_class in self.provider_map.items():
            if base_class in class_hierarchy:
                model_provider = provider
                break
        
        model_id = serialized.get("kwargs", {}).get("model")

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
            self._run_info[run_id] = {"provider": model_provider.value, "model_name": model_id}

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            run_info = self._run_info.pop(run_id, None)

        if not run_info:
            if self.logger.isEnabledFor(logging.DEBUG):
                 self.logger.debug(f"No start information for run {run_id}. This is normal for fake models.")
            return

        provider = run_info["provider"]
        model_name = run_info["model_name"]
        
        if not response.llm_output or "token_usage" not in response.llm_output:
            self.logger.warning(f"No token_usage information in LLM output for run {run_id}. Cost cannot be calculated.")
            return

        token_usage = response.llm_output.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        run_cost = 0.0
        guardrails_tokens = 0

        try:
            model_costs = self.cost_dict[provider][model_name]
            
            if provider == Provider.SNOWFLAKE:
                guardrails_tokens = token_usage.get("guardrails_tokens", 0)
                total_run_tokens = prompt_tokens + completion_tokens + guardrails_tokens
                
                cost_per_mil = model_costs.get("cost", 0.0)
                run_cost = (total_run_tokens / 1_000_000) * cost_per_mil

            else:
                input_cost_per_mil = model_costs.get("input_cost", 0.0)
                output_cost_per_mil = model_costs.get("output_cost", 0.0)
                run_cost = (prompt_tokens / 1_000_000 * input_cost_per_mil) + \
                           (completion_tokens / 1_000_000 * output_cost_per_mil)

        except KeyError:
            self.logger.warning(
                f"Cost not found for model '{model_name}' from provider '{provider}'. "
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