from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple


class BudgetAgent:
    """Wrapper around backend/packages/budget-bot.py with a clean interface."""

    def __init__(self, module_dir: Path | None = None):
        base_dir = module_dir or Path(__file__).parent.resolve()
        self._module_path = (base_dir / "budget-bot.py").resolve()
        self._mod = self._load_module()

    def _load_module(self):
        spec = importlib.util.spec_from_file_location(
            "budget_bot", str(self._module_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load budget-bot module from {self._module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    # Public methods
    def call(self, user_input: str, recursion_limit: int = 25) -> Tuple[str, Any]:
        return self._mod.call_budget_agent_tools(user_input, recursion_limit)

    def get_context(self) -> Dict[str, Any]:
        return self._mod.get_budget_context()

    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._mod.set_budget_context(ctx)

    # Convenience accessors
    @property
    def memory_path(self) -> str:
        return getattr(self._mod, "MEMORY_PATH", "")

    @property
    def faiss_path(self) -> str:
        return getattr(self._mod, "BUDGET_FAISS_PATH", "")
