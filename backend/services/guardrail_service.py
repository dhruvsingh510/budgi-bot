import os
from pathlib import Path
from typing import Optional

from logger_config import get_service_logger
from guardrails import FinanceDomainGuard, GuardDecision


class GuardrailService:
    """Wrapper around FinanceDomainGuard mirroring other service patterns."""

    def __init__(
        self,
        llm,
        config_path: Optional[str] = None,
        logger=None,
        config_section_key: str = "finance_domain_guard",
    ) -> None:
        self.logger = logger or get_service_logger("guardrail")
        self.config_path = config_path or os.path.abspath(
            os.path.join(
                Path(__file__).parent, "..", "guardrails", "finance_guard.yaml"
            )
        )
        self.config_section_key = config_section_key
        self.guard = FinanceDomainGuard(
            llm=llm,
            logger=self.logger,
            config_path=self.config_path,
            config_section_key=self.config_section_key,
        )
        self.logger.info(
            "GuardrailService initialized. config=%s section=%s enabled=%s",
            self.config_path,
            self.config_section_key,
            self.guard.enabled,
        )

    @property
    def fallback_response(self) -> str:
        return self.guard.fallback_response

    @property
    def enabled(self) -> bool:
        return self.guard.enabled

    def evaluate(self, user_input: str, assistant_output: str) -> GuardDecision:
        sanitized_input = (user_input or "").replace("\n", " ").strip()
        display_input = sanitized_input[:200] + (
            "..." if len(sanitized_input) > 200 else ""
        )
        self.logger.info(
            "Evaluating guardrail (section=%s) for user_input='%s'",
            self.config_section_key,
            display_input,
        )
        decision = self.guard.evaluate(user_input, assistant_output)
        self.logger.info(
            "Guardrail decision (section=%s): allowed=%s reason=%s user_input='%s'",
            self.config_section_key,
            decision.allowed,
            decision.reason,
            display_input,
        )
        return decision
