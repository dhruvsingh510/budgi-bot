import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from logger_config import get_service_logger
from guardrail_service import GuardrailService
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


@dataclass
class BudgetProfile:
    net_income: float  # net income
    cost_of_living: str  # high, medium, low
    household_size: int  # no. of people in household


@dataclass
class FinancialGoal:
    name: str  # name of financial goal
    target_amount: float  # amount to be saved
    months: int  # no. of months required to save amount
    priority: int  # 1 - 5, 1 is highest


class BudgetService:
    """Budget service with dependency injection."""

    # Constants
    ADVICE_KEYWORDS = (
        "would you like",
        "tips on",
        "provide tips",
        "consider",
        "you should",
        "try to",
        "recommend",
        "suggest",
        "ways to reduce",
        "increase your savings",
        "explore ways",
        "discuss how",
        "help you achieve",
    )

    CATEGORIES = [
        "Housing & Utilities",
        "Food & Groceries",
        "Transportation",
        "Insurance",
        "Healthcare",
        "Debt & Loans",
        "Savings & Investments",
        "Entertainment & Leisure",
        "Shopping",
        "Education",
        "Personal Care",
        "Travel & Holidays",
        "Family & Childcare",
        "Miscellaneous",
    ]

    GUIDE_DEFAULTS = {
        "high": {
            "Housing & Utilities": (35, 35),
            "Food & Groceries": (10, 10),
            "Transportation": (7, 7),
            "Insurance": (5, 5),
            "Healthcare": (5, 5),
            "Debt & Loans": (5, 5),
            "Savings & Investments": (10, 10),
            "Entertainment & Leisure": (6, 6),
            "Shopping": (5, 5),
            "Education": (3, 3),
            "Personal Care": (3, 3),
            "Travel & Holidays": (3, 3),
            "Family & Childcare": (2, 2),
            "Miscellaneous": (1, 1),
        },
        "medium": {
            "Housing & Utilities": (30, 30),
            "Food & Groceries": (12, 12),
            "Transportation": (9, 9),
            "Insurance": (5, 5),
            "Healthcare": (5, 5),
            "Debt & Loans": (5, 5),
            "Savings & Investments": (10, 10),
            "Entertainment & Leisure": (6, 6),
            "Shopping": (5, 5),
            "Education": (4, 4),
            "Personal Care": (3, 3),
            "Travel & Holidays": (3, 3),
            "Family & Childcare": (2, 2),
            "Miscellaneous": (1, 1),
        },
        "low": {
            "Housing & Utilities": (25, 25),
            "Food & Groceries": (12, 12),
            "Transportation": (7, 7),
            "Insurance": (5, 5),
            "Healthcare": (4, 4),
            "Debt & Loans": (6, 6),
            "Savings & Investments": (12, 12),
            "Entertainment & Leisure": (6, 6),
            "Shopping": (6, 6),
            "Education": (4, 4),
            "Personal Care": (4, 4),
            "Travel & Holidays": (4, 4),
            "Family & Childcare": (4, 4),
            "Miscellaneous": (1, 1),
        },
    }

    ELASTICITY_ORDER = [
        "Entertainment & Leisure",
        "Shopping",
        "Travel & Holidays",
        "Miscellaneous",
        "Personal Care",
        "Education",
        "Transportation",
        "Food & Groceries",
        "Healthcare",
    ]

    FLOORS = {"Food & Groceries": 8, "Healthcare": 3}

    def __init__(
        self,
        llm: ChatGroq,
        text_splitter: RecursiveCharacterTextSplitter,
        embeddings: HuggingFaceEmbeddings,
        vector_store: Optional[FAISS],
        memory_path: str,
    ):
        self.llm = llm
        self.text_splitter = text_splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.memory_path = memory_path

        # Initialize logger first
        self.logger = get_service_logger("budget")
        self.response_guard = GuardrailService(
            llm=self.llm,
            config_path=os.path.abspath(
                os.path.join(
                    Path(__file__).parent,
                    "..",
                    "guardrails",
                    "budget_response_guard.yaml",
                )
            ),
            config_section_key="budget_response_guard",
            logger=get_service_logger("budget_guard"),
        )
        self.last_guard_state: Dict[str, Any] = {}

        # Load context from memory
        self.context = self._load_memory()

        # Initialize the LangGraph workflow
        self._setup_workflow()
        self.logger.info("Budget service initialized")

    def _load_memory(self) -> Dict[str, Any]:
        """Load budget context from persistent storage."""
        try:
            with open(self.memory_path, "r") as f:
                data = json.load(f)
            # reconstruct objects
            profile = data.get("profile")
            if profile:
                profile = BudgetProfile(**profile)
            goals = [FinancialGoal(**g) for g in data.get("goals", [])]
            plan = data.get("plan")
            loaded_budget_profile = {"profile": profile, "goals": goals, "plan": plan}
            self.logger.info(f"Loaded memory: {loaded_budget_profile}")
            return loaded_budget_profile
        except Exception:
            self.logger.error("Failed to load memory")
            return {"profile": None, "goals": [], "plan": None}

    def _save_memory(self, ctx: Dict[str, Any]) -> None:
        """Save budget context to persistent storage."""
        try:
            serializable = {
                "profile": ctx["profile"].__dict__ if ctx.get("profile") else None,
                "goals": [g.__dict__ for g in ctx.get("goals", [])],
                "plan": ctx.get("plan"),
                "updated_at": datetime.now(
                    timezone(timedelta(hours=5, minutes=30))
                ).isoformat(),
            }
            with open(self.memory_path, "w") as f:
                json.dump(serializable, f, indent=2)
            self.logger.info(f"Saved memory: {serializable}")
        except Exception as e:
            self.logger.error(f"Warning: failed saving memory: {e}")

    # Budget calculation logic
    @staticmethod
    def midpoint(a: int, b: int) -> float:
        return (a + b) / 2.0

    def propose_base_percentages(self, profile: BudgetProfile) -> Dict[str, float]:
        guide = self.GUIDE_DEFAULTS.get(
            profile.cost_of_living, self.GUIDE_DEFAULTS["medium"]
        )
        base = {cat: self.midpoint(*guide[cat]) for cat in self.CATEGORIES}
        return base

    def monthly_goal_savings(self, goals: List[FinancialGoal]) -> float:
        return sum(g.target_amount / max(g.months, 1) for g in goals)

    def allocate_budget(
        self, profile: BudgetProfile, goals: List[FinancialGoal]
    ) -> Dict[str, Any]:
        base_pct = self.propose_base_percentages(profile)
        total_base = sum(base_pct.values())

        # If percentages exceed 100%, scale them down proportionally
        scale = min(100.0 / max(total_base, 1e-6), 1.0)
        for k in base_pct:
            base_pct[k] *= scale

        # Reserve savings for goals by reducing variable categories per elasticity
        income = profile.net_income
        required_savings = self.monthly_goal_savings(goals)

        # Convert to amounts
        allocations = {k: income * (p / 100.0) for k, p in base_pct.items()}

        # If needed, carve out savings
        if required_savings > 0:
            carved = 0.0
            for cat in self.ELASTICITY_ORDER:
                if carved >= required_savings:
                    break
                floor_pct = self.FLOORS.get(cat, 0.0)
                floor_amt = income * (floor_pct / 100.0)
                available = max(allocations[cat] - floor_amt, 0.0)
                take = min(required_savings - carved, available)
                allocations[cat] -= take
                carved += take

            savings_shortfall = max(required_savings - carved, 0.0)
        else:
            savings_shortfall = 0.0

        plan = {
            "allocations": allocations,
            "required_savings": required_savings,
            "savings_shortfall": savings_shortfall,
            "income": income,
        }
        self.logger.info(f"Allocated budget: {plan}")
        return plan

    def render_plan(self, plan: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"Monthly Net Income: ₹{plan['income']:,.2f}")
        lines.append("Proposed Monthly Allocations:")
        for cat in self.CATEGORIES:
            amt = plan["allocations"][cat]
            pct = (amt / max(plan["income"], 1e-6)) * 100
            lines.append(f"- {cat}: ₹{amt:,.2f} ({pct:.1f}%)")
        if plan["required_savings"] > 0:
            lines.append(f"Savings toward goals: ₹{plan['required_savings']:,.2f}")
            if plan["savings_shortfall"] > 0:
                lines.append(
                    f"Shortfall: ₹{plan['savings_shortfall']:,.2f} — consider extending deadlines or tightening discretionary spend."
                )
        return "\n".join(lines)

    # Utility parsers
    @staticmethod
    def _parse_float(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        s = s.replace(",", " ")
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not nums:
            raise ValueError(f"Could not parse float from: {value}")
        return float(nums[0])

    @staticmethod
    def _parse_int(value) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        nums = re.findall(r"-?\d+", str(value))
        if not nums:
            raise ValueError(f"Could not parse int from: {value}")
        return int(nums[0])

    @staticmethod
    def _parse_months(value) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip().lower()
        # handle year-based durations
        m = re.search(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|y)\b", s)
        if m:
            return int(round(float(m.group(1)) * 12))
        # handle month-based durations
        m = re.search(r"(\d+)\s*(months?|mos?|m)\b", s)
        if m:
            return int(m.group(1))
        # fallback: first integer found
        return BudgetService._parse_int(value)

    def _get_system_prompt(self) -> str:
        return """
        You are a budgeting assistant with persistent episodic memory saved on disk.
        - Only call a tool if strictly necessary to read/change state (profile, goals, plan) or to compute/propose/adjust.
        - After a successful tool call, reply to the user in natural language and DO NOT call any more tools in this turn.

        IMP : You can not advice any strategies or recommendations for budgeting or reaching target.
        You should keep your responses short, meaningful, and also professional.
        Try to use simple language that is easily understandable, and not wordy.

        All financial amounts should be displayed in Indian Rupees (INR) using the ₹ symbol.
        When setting profiles, users provide annual income which is automatically converted to monthly income for budget calculations.

        DO NOT REVEAL EXTRA INFORMATION OF YOURSELF SO PEOPLE CAN TELL THAT YOU ARE A BOT.
        DO NOT RECOMMEND ANY NEXT STEPS, OR GIVE RESPONSES THAT REQUIRE MORE USER INPUT.
        KEEP MESSAGES SHORT AND CONCISE. DON'T GIVE VERY LONG RESPONSES.
        DO NOT GIVE ANY INFORMATION RELATED TO THE CALCULATION OF THE ALLOCATIONS.
        """

    def process_request(
        self, user_input: str, recursion_limit: int = 25
    ) -> Tuple[str, Any]:
        """Process user request and return response."""
        state = {
            "messages": [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=user_input),
            ]
        }
        final = self.tool_app.invoke(
            state, config={"recursion_limit": recursion_limit, "debug": True}
        )
        ai_msgs = [m for m in final["messages"] if isinstance(m, AIMessage)]
        text = ai_msgs[-1].content if ai_msgs else "Done."

        stripped_text = text.strip()
        CLEAR_CONFIRMATIONS = {
            "Cleared all goals and the current plan.",
            "Cleared current budget plan.",
            "Cleared profile, and plan.",
        }
        if stripped_text in CLEAR_CONFIRMATIONS:
            self.logger.info(
                "Bypassing guard for clear operation response: %s", stripped_text
            )
            self.last_guard_state = {"allowed": True, "reason": "clear_operation"}
            return stripped_text, final

        guard_decision = self.response_guard.evaluate(user_input, text)
        self.last_guard_state = {
            "allowed": guard_decision.allowed,
            "reason": guard_decision.reason,
        }
        needs_rewrite = False
        if not guard_decision.allowed:
            needs_rewrite = True
            self.logger.warning(
                "Budget response guard blocked output. reason=%s",
                guard_decision.reason,
            )
        elif self._contains_advice(text):
            needs_rewrite = True
            self.logger.info(
                "Budget response contained advisory language; triggering rewrite."
            )

        if needs_rewrite:
            plan_summary = None
            plan = self.context.get("plan")
            if plan:
                plan_summary = self.render_plan(plan)

            profile_summary = None
            profile = self.context.get("profile")
            if profile:
                profile_summary = self._render_profile_summary(profile)

            rewritten = self._rewrite_without_advice(user_input, text)
            if rewritten and not self._contains_advice(rewritten):
                self.logger.info("Budget response guard produced rewritten response.")
                sections = [rewritten.strip()] if rewritten.strip() else []
                if profile_summary:
                    sections.append(profile_summary)
                if plan_summary and plan_summary not in sections:
                    sections.append(plan_summary)
                text = "\n\n".join(sections)
            elif plan_summary:
                self.logger.info(
                    "Budget response guard falling back to rendered plan summary."
                )
                sections = []
                if profile_summary:
                    sections.append(profile_summary)
                sections.append(plan_summary)
                text = "\n\n".join(sections)
            else:
                self.logger.warning(
                    "Budget response guard rewrite failed; using fallback response."
                )
                text = self.response_guard.fallback_response

        return text, final

    def get_context(self) -> Dict[str, Any]:
        """Get current budget context."""
        self.logger.info(f"Getting context: {self.context}")
        return self.context

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set budget context."""
        self.context.clear()
        self.context.update(
            {
                "profile": context.get("profile"),
                "goals": context.get("goals", []),
                "plan": context.get("plan"),
            }
        )
        self.logger.info(f"Set context: {self.context}")
        self._save_memory(self.context)

    def _setup_workflow(self):
        """Setup the LangGraph workflow with tools."""

        # Create tools that operate on self.context
        @tool
        def set_profile_tool(
            net_income: str, cost_of_living: str, household_size: str
        ) -> str:
            """Set the user's profile: net annual income in INR (will be converted to monthly), cost_of_living in {low, medium, high}, and household size."""
            # Store as monthly income for internal calculations (annual / 12)
            annual_income = self._parse_float(net_income)
            monthly_income = annual_income / 12

            profile = BudgetProfile(
                net_income=monthly_income,
                cost_of_living=str(cost_of_living).lower(),
                household_size=self._parse_int(household_size),
            )
            self.context["profile"] = profile
            self._save_memory(self.context)
            self.logger.info(
                f"Profile saved: annual_income={annual_income}, monthly_income={monthly_income}, profile={profile}"
            )
            return f"Profile saved successfully! Annual income: ₹{annual_income:,.0f}, Monthly income: ₹{monthly_income:,.2f}, Cost of living: {cost_of_living.title()}, Household size: {household_size}. You can now ask to add a financial goal or propose budget."

        @tool
        def add_goal_tool(
            name: str, target_amount: str, months: str, priority: str = "3"
        ) -> str:
            """Add a financial goal with a name, target_amount, months to reach, and priority (1 highest and 5 lowest)."""
            goal = FinancialGoal(
                name=name,
                target_amount=self._parse_float(target_amount),
                months=self._parse_months(months),
                priority=self._parse_int(priority),
            )
            self.context.setdefault("goals", []).append(goal)

            profile = self.context.get("profile")
            if self.context.get("plan") is not None and profile is not None:
                plan = self.allocate_budget(profile, self.context.get("goals", []))
                self.context["plan"] = plan
                self._save_memory(self.context)
                return (
                    f"{self.render_plan(plan)}"
                    + "\nFirst, tell the user that the requested goal was added. Secondly, on the basis of the updated goals, a new plan was created. Inform the user about both the things"
                )

            self._save_memory(self.context)
            self.logger.info(f"Financial goal added: {goal}")
            return f"Added financial goal '{name}'."

        @tool
        def propose_budget_tool() -> str:
            """Compute a proposed monthly budget based on the saved profile and goals."""
            profile = self.context.get("profile")
            goals = self.context.get("goals", [])
            if profile is None:
                return "Please set your profile first."
            plan = self.allocate_budget(profile, goals)
            self.context["plan"] = plan
            self._save_memory(self.context)
            self.logger.info(f"Proposed budget: {plan}")
            return self.render_plan(plan)

        @tool
        def adjust_category_tool(category: str, amount: str) -> str:
            """Adjust a category amount in the current plan."""
            plan = self.context.get("plan")
            if plan is None:
                return "No plan yet. Ask to 'propose budget' first."
            if category not in self.CATEGORIES:
                return f"Unknown category. Choose from: {', '.join(self.CATEGORIES)}."
            plan["allocations"][category] = max(self._parse_float(amount), 0.0)
            self.context["plan"] = plan
            self._save_memory(self.context)
            self.logger.info(f"Adjusted category: {category} to {amount}")
            return self.render_plan(plan)

        @tool
        def clear_goals_tool() -> str:
            """Remove all currently saved goals and any existing plan."""
            self.context["goals"] = []
            self.context["plan"] = None
            self._save_memory(self.context)
            self.logger.info("Cleared all goals and the current plan.")
            return "Cleared all goals and the current plan."

        @tool
        def clear_plan_tool() -> str:
            """Remove the current budget plan."""
            self.context["plan"] = None
            self._save_memory(self.context)
            self.logger.info("Cleared current budget plan.")
            return "Cleared current budget plan."

        @tool
        def clear_profile_tool() -> str:
            """Remove the saved profile and any dependent state (goals and plan)."""
            self.context["profile"] = None
            self.context["plan"] = None
            self._save_memory(self.context)
            self.logger.info("Cleared profile, and plan.")
            return "Cleared profile, and plan."

        @tool
        def show_plan_tool() -> str:
            """Show the current budget plan if available."""
            plan = self.context.get("plan")
            if not plan:
                return "No plan yet. Ask to 'propose budget' first."
            self.logger.info(f"Showing plan: {plan}")
            return self.render_plan(plan)

        @tool
        def show_goals_tool() -> str:
            """List all saved goals from memory."""
            goals = self.context.get("goals", [])
            if not goals:
                return "No goals saved yet. Please create a 'financial goal' first."
            lines = ["Saved Financial Goals:"]
            for g in goals:
                lines.append(
                    f"- {g.name}: target ₹{g.target_amount:,.2f} in {g.months} months (priority {g.priority})"
                )
            self.logger.info(f"Showing goals: {lines}")
            return "\n".join(lines)

        @tool
        def show_profile_tool() -> str:
            """Show the current saved profile."""
            p = self.context.get("profile")
            if not p:
                return "No profile saved yet. Please create a 'profile' first."
            self.logger.info(f"Showing profile: {p}")
            annual_income = p.net_income * 12
            return f"Current Profile:\n- Annual Income: ₹{annual_income:,.0f}\n- Monthly Income: ₹{p.net_income:,.2f}\n- Cost of Living: {p.cost_of_living.title()}\n- Household Size: {p.household_size}"

        self.tools = [
            set_profile_tool,
            add_goal_tool,
            propose_budget_tool,
            adjust_category_tool,
            clear_goals_tool,
            clear_plan_tool,
            clear_profile_tool,
            show_plan_tool,
            show_goals_tool,
            show_profile_tool,
        ]

        # Message-based state for ToolNode flow
        class MsgState(TypedDict):
            messages: List
            tool_calls_made: int

        def _format_memory_for_prompt() -> str:
            p = self.context.get("profile")
            goals = self.context.get("goals", [])
            plan = self.context.get("plan")
            parts = [
                "You have access to persistent episodic memory. Current saved state:"
            ]
            if p:
                parts.append(
                    f"- Profile: net_income={p.net_income}, cost_of_living={p.cost_of_living}, household_size={p.household_size}"
                )
            if goals:
                parts.append("- Goals:")
                for g in goals:
                    parts.append(
                        f"  • {g.name}: target {g.target_amount} in {g.months} months (priority {g.priority})"
                    )
            if plan:
                parts.append("- A budget plan exists for the current profile/goals.")
            if not p and not goals and not plan:
                parts.append("- No saved profile, goals, or plan.")
            parts.append("Use this memory as context when reasoning and responding.")
            return "\n".join(parts)

        # Model node that can request tools (disabled after first tool call)
        def budget_model_node(state: MsgState) -> MsgState:
            tools_allowed = state.get("tool_calls_made", 0) < 1
            model = self.llm if not tools_allowed else self.llm.bind_tools(self.tools)
            # Inject memory summary for model context without mutating the persisted chat history
            messages_to_model = list(state["messages"]) or []
            try:
                mem_msg = SystemMessage(content=_format_memory_for_prompt())
                # Insert after the first SystemMessage if present, else prepend
                inserted = False
                for i, m in enumerate(messages_to_model):
                    if isinstance(m, SystemMessage):
                        messages_to_model.insert(i + 1, mem_msg)
                        inserted = True
                        break
                if not inserted:
                    messages_to_model = [mem_msg] + messages_to_model
            except Exception:
                pass
            ai = model.invoke(messages_to_model)
            final_state = {
                "messages": state["messages"] + [ai],
                "tool_calls_made": state.get("tool_calls_made", 0),
            }
            self.logger.info(f"Budget Model Node State: {final_state}")
            return final_state

        # Prebuild a tool runner instance
        _tools_runner = ToolNode(self.tools)

        # Tools node that executes tools and increments the counter
        def tools_node(state: MsgState) -> MsgState:
            result = _tools_runner.invoke(state)
            final_state = {
                "messages": result["messages"],
                "tool_calls_made": state.get("tool_calls_made", 0) + 1,
            }
            self.logger.info(f"Budget Tools Node State: {final_state}")
            return final_state

        # Routing: if the last AI message has tool calls, go to tools; else end
        def route_tools(state: MsgState):
            if state.get("tool_calls_made", 0) >= 1:
                return END
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        # Build tool-enabled graph
        tool_graph = StateGraph(MsgState)
        tool_graph.add_node("model", budget_model_node)
        tool_graph.add_node("tools", tools_node)
        tool_graph.add_edge(START, "model")
        tool_graph.add_conditional_edges("model", route_tools)
        tool_graph.add_edge("tools", "model")
        self.tool_app = tool_graph.compile()

    def _rewrite_without_advice(
        self, user_input: str, assistant_output: str
    ) -> Optional[str]:
        """Rewrite assistant output to remove advice while keeping factual info."""
        system_prompt = (
            "You rewrite budget assistant responses so they stay factual, concise, and "
            "avoid offering advice, suggestions, or calls to action. Return only the rewritten response."
        )
        human_prompt = f"""
User request:
{user_input}

Original budget response:
{assistant_output}

Rewrite the response so that it:
- Summarizes the user's budget/profile/plan facts that were provided.
- Does NOT include advice, suggestions, tips, or instructions.
- Keeps the tone professional and concise.
- Does NOT ask follow-up questions or offer next steps.

Provide the rewritten response as plain text.
"""
        try:
            rewritten = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt.strip()),
                ]
            )
            return (
                rewritten.content.strip() if rewritten and rewritten.content else None
            )
        except Exception as exc:
            self.logger.error(
                "Failed to rewrite budget response without advice: %s",
                exc,
                exc_info=True,
            )
            return None

    @classmethod
    def _contains_advice(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return any(keyword in lowered for keyword in cls.ADVICE_KEYWORDS)

    @staticmethod
    def _render_profile_summary(profile: BudgetProfile) -> str:
        return (
            "Current Profile:\n"
            f"- Annual Income: ₹{profile.net_income * 12:,.0f}\n"
            f"- Monthly Income: ₹{profile.net_income:,.2f}\n"
            f"- Cost of Living: {profile.cost_of_living.title()}\n"
            f"- Household Size: {profile.household_size}"
        )
