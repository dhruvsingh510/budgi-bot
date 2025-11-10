import json
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml
from langchain_core.messages import HumanMessage

try:
    from nemoguardrails import RailsConfig  # type: ignore[import]
except Exception:  # pragma: no cover - Guards against missing dependency
    RailsConfig = None  # type: ignore[assignment]


@dataclass
class GuardDecision:
    """Result of a guardrail evaluation."""

    allowed: bool
    reason: Optional[str] = None


class FinanceDomainGuard:
    """Evaluate if conversations remain within finance-related topics."""

    DEFAULT_FALLBACK = (
        "I can help with budgeting, transactions, or general finance questions. "
        "Could you please ask something along those lines?"
    )

    def __init__(
        self,
        llm,
        logger,
        config_path: Optional[str] = None,
        config_section_key: str = "finance_domain_guard",
    ) -> None:
        self._llm = llm
        self._logger = logger
        self._config_path = config_path
        self._config_section_key = config_section_key
        self._enabled = False
        self._prompt_template: Optional[str] = None
        self._allowed_topics: List[str] = [
            "budgeting",
            "budget",
            "transaction",
            "transactions",
            "general finance",
            "greetings",
        ]
        self._fallback_response: str = self.DEFAULT_FALLBACK

        self._load_configuration()

    @property
    def enabled(self) -> bool:
        """Whether guardrail evaluations are active."""
        return self._enabled

    @property
    def fallback_response(self) -> str:
        """Response to return when guardrail blocks an answer."""
        return self._fallback_response

    def _load_configuration(self) -> None:
        """Load guard configuration from YAML using NeMo guardrails if available."""
        if not self._config_path or not os.path.exists(self._config_path):
            self._logger.warning(
                "FinanceDomainGuard config not found at %s; using defaults.",
                self._config_path,
            )
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                raw_config = fh.read()
        except Exception as ex:  # pragma: no cover - defensive fallback
            self._logger.error(
                "Failed to read finance guard configuration: %s", ex, exc_info=True
            )
            return

        try:
            parsed_yaml = yaml.safe_load(raw_config) or {}
        except Exception as ex:  # pragma: no cover - defensive fallback
            self._logger.error(
                "Unable to parse finance guard configuration: %s", ex, exc_info=True
            )
            return

        guard_section = parsed_yaml.get(self._config_section_key, {})
        prompt_template = guard_section.get("prompt_template")
        allowed_topics = guard_section.get("allowed_topics")
        fallback_response = guard_section.get("fallback_response")

        if prompt_template:
            self._prompt_template = prompt_template
        if isinstance(allowed_topics, list) and allowed_topics:
            self._allowed_topics = [str(topic).lower() for topic in allowed_topics]
        if isinstance(fallback_response, str) and fallback_response.strip():
            self._fallback_response = fallback_response.strip()

        # Attempt to validate configuration through NeMo guardrails metadata parsing.
        if RailsConfig is not None:
            try:
                RailsConfig.from_content(raw_config)
            except Exception as ex:
                self._logger.warning(
                    "NeMo guardrails validation failed: %s. Proceeding with YAML-only "
                    "configuration.",
                    ex,
                )

        if not self._prompt_template:
            self._logger.warning(
                "FinanceDomainGuard missing prompt template for section '%s'; guard will remain disabled.",
                self._config_section_key,
            )
            return

        if self._llm is None:
            self._logger.warning(
                "FinanceDomainGuard requires an initialized LLM; guard disabled."
            )
            return

        self._enabled = True
        self._logger.info(
            "FinanceDomainGuard initialized for section '%s' with topics: %s",
            self._config_section_key,
            self._allowed_topics,
        )

    def evaluate(self, user_input: str, assistant_output: str) -> GuardDecision:
        """Run the guardrail evaluation."""
        if not self._enabled:
            self._logger.debug(
                "FinanceDomainGuard skipped (disabled). Input='%s'", user_input
            )
            return GuardDecision(allowed=True, reason="guard_disabled")

        template = self._prompt_template or ""
        rendered_prompt = template.format(
            allowed_topics=", ".join(self._allowed_topics),
            user_input=user_input.strip(),
            assistant_output=assistant_output.strip(),
        )

        sanitized_input = user_input.strip()[:200]
        assistant_len = len(assistant_output or "")
        self._logger.info(
            "FinanceDomainGuard evaluating (section=%s). user_input='%s', assistant_output_len=%d",
            self._config_section_key,
            sanitized_input + ("..." if len(user_input.strip()) > 200 else ""),
            assistant_len,
        )

        try:
            response = self._llm.invoke([HumanMessage(content=rendered_prompt)])
        except Exception as ex:
            self._logger.error(
                "FinanceDomainGuard invocation failed: %s", ex, exc_info=True
            )
            return GuardDecision(allowed=True, reason="guard_invoke_error")

        raw_content = (response.content or "").strip()
        self._logger.debug(
            "FinanceDomainGuard raw model response (section=%s): %s",
            self._config_section_key,
            raw_content[:500],
        )

        if not raw_content:
            self._logger.error(
                "FinanceDomainGuard received empty response from model (section=%s)",
                self._config_section_key,
            )
            return GuardDecision(allowed=True, reason="guard_empty_response")

        parsed = None
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            normalized = raw_content.replace("-", "_").strip().lower()
            if normalized in {"on_topic", "on topic", "ontopic", "on"}:
                parsed = {"is_on_topic": True, "reason": "string_on_topic"}
            elif normalized in {"off_topic", "off topic", "offtopic", "off"}:
                parsed = {"is_on_topic": False, "reason": "string_off_topic"}
            else:
                try:
                    parsed_yaml = yaml.safe_load(raw_content)
                    if isinstance(parsed_yaml, dict):
                        parsed = parsed_yaml
                except Exception:
                    parsed = None

        if parsed is None:
            self._logger.error(
                "FinanceDomainGuard could not parse model response as JSON/YAML (section=%s)",
                self._config_section_key,
            )
            return GuardDecision(allowed=True, reason="guard_parse_error")

        is_on_topic = bool(parsed.get("is_on_topic", False))
        reason = parsed.get("reason")

        if not is_on_topic and reason:
            reason = str(reason)

        decision = GuardDecision(allowed=is_on_topic, reason=reason)
        self._logger.info(
            "FinanceDomainGuard decision (section=%s): allowed=%s, reason=%s",
            self._config_section_key,
            decision.allowed,
            decision.reason,
        )

        return decision
