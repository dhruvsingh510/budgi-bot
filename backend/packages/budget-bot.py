# Importing all packages
from langgraph.graph import StateGraph, START, END
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass

import os
import re
import json


# Resolve paths relative to this file so imports work regardless of CWD
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BUDGET_FAISS_PATH = os.path.abspath(
    os.path.join(MODULE_DIR, "..", "data", "faiss_store_budget")
)
MEMORY_PATH = os.path.abspath(
    os.path.join(MODULE_DIR, "..", "data", "budget_memory.json")
)


# Loading env variables
load_dotenv()


# Splitting of sample data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
# Setting up embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Dataclasses for setting budget profile and financial goal
@dataclass
class BudgetProfile:
    net_income: float  # net income
    cost_of_living: str  # high, medium, low
    household_size: int  # no. of people in household


@dataclass
class FinancialGoal:
    name: str  # name of financial goal
    target_amount: float  # amount to be saved
    months: int  # no. of months reuired to save amount
    priority: int  # 1 - 5, 1 is highest


# Create/Load guidelines store
if os.path.exists(f"{BUDGET_FAISS_PATH}/index.faiss"):
    db_guidelines = FAISS.load_local(
        f"{BUDGET_FAISS_PATH}",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("✅ Loaded existing budget guidelines FAISS")
else:
    print("🔴 Unable to fetch budget guidelines FAISS")


# Initialise budget_context from memory
def _load_memory():
    try:
        with open(MEMORY_PATH, "r") as f:
            data = json.load(f)
        # reconstruct objects
        profile = data.get("profile")
        if profile:
            profile = BudgetProfile(**profile)
        goals = [FinancialGoal(**g) for g in data.get("goals", [])]
        plan = data.get("plan")
        return {"profile": profile, "goals": goals, "plan": plan}
    except Exception:
        return {"profile": None, "goals": [], "plan": None}


budget_context = _load_memory()


# Save budget_context to memory
def _save_memory(ctx):
    try:
        serializable = {
            "profile": ctx["profile"].__dict__ if ctx.get("profile") else None,
            "goals": [g.__dict__ for g in ctx.get("goals", [])],
            "plan": ctx.get("plan"),
            "updated_at": datetime.now(
                timezone(timedelta(hours=5, minutes=30))
            ).isoformat(),
        }
        with open(MEMORY_PATH, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"Warning: failed saving memory: {e}")


# Initialize LLM
LLM = ChatGroq(
    model_name=os.environ.get("LITELLM_MODEL"),
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)

# TO DO : This can also be set from the transaction_memory.json file for fetching newly created categories
CATEGORIES = [
    "Income",
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


# Simple rules to convert guideline text into midpoints
GUIDE_DEFAULTS = {
    "high": {
        "Income": (0, 0),
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
        "Income": (0, 0),
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
        "Income": (0, 0),
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

# When carving out savings or rebalancing, cut in this order (most flexible first)
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

# Minimum percentages that should not be cut below (as % of income)
FLOORS = {"Food & Groceries": 8, "Healthcare": 3}


def midpoint(a: int, b: int) -> float:
    return (a + b) / 2.0


def propose_base_percentages(profile: BudgetProfile) -> Dict[str, float]:
    guide = GUIDE_DEFAULTS.get(profile.cost_of_living, GUIDE_DEFAULTS["medium"])
    base = {cat: midpoint(*guide[cat]) for cat in CATEGORIES}
    return base


def monthly_goal_savings(goals: List[FinancialGoal]) -> float:
    return sum(g.target_amount / max(g.months, 1) for g in goals)


def allocate_budget(
    profile: BudgetProfile, goals: List[FinancialGoal]
) -> Dict[str, Any]:
    base_pct = propose_base_percentages(
        profile
    )  # Get percentages based on cost of living
    total_base = sum(base_pct.values())  # Add them up

    # If percentages exceed 100%, scale them down proportionally
    scale = min(100.0 / max(total_base, 1e-6), 1.0)
    for k in base_pct:
        base_pct[k] *= scale

    # Reserve savings for goals by reducing variable categories per elasticity
    income = profile.net_income
    required_savings = monthly_goal_savings(goals)

    # Convert to amounts
    allocations = {k: income * (p / 100.0) for k, p in base_pct.items()}

    # If needed, carve out savings
    if required_savings > 0:
        carved = 0.0
        for cat in ELASTICITY_ORDER:
            # Stop if we've already carved out enough money for all goals
            if carved >= required_savings:
                break
            # Check if this category has a minimum floor (e.g., Food & Groceries >= 8%)
            floor_pct = FLOORS.get(cat, 0.0)
            floor_amt = income * (floor_pct / 100.0)
            # Calculate how much we can safely cut from this category
            available = max(allocations[cat] - floor_amt, 0.0)
            # Take either what we still need OR what's available, whichever is smaller
            take = min(required_savings - carved, available)
            # Actually reduce the category allocation by the amount we're taking
            allocations[cat] -= take
            # Keep track of total amount carved out across all categories
            carved += take

        # Calculate any shortfall: if we couldn't carve out enough due to floors
        # (e.g., goals too aggressive, hit minimum spending requirements)
        savings_shortfall = max(required_savings - carved, 0.0)
    else:
        savings_shortfall = 0.0

    plan = {
        "allocations": allocations,
        "required_savings": required_savings,
        "savings_shortfall": savings_shortfall,
        "income": income,
    }
    return plan


def render_plan(plan: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Net income: {plan['income']:.2f}")
    lines.append("Proposed monthly allocations:")
    for cat in CATEGORIES:
        amt = plan["allocations"][cat]
        pct = (amt / max(plan["income"], 1e-6)) * 100
        lines.append(f"- {cat}: {amt:.2f} ({pct:.1f}%)")
    if plan["required_savings"] > 0:
        lines.append(f"Savings toward goals: {plan['required_savings']:.2f}")
        if plan["savings_shortfall"] > 0:
            lines.append(
                f"Shortfall: {plan['savings_shortfall']:.2f} — consider extending deadlines or tightening discretionary spend."
            )
    return "\n".join(lines)

    # Tools that operate on budget_context


def _parse_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    # remove currency symbols and commas
    s = s.replace(",", " ")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        raise ValueError(f"Could not parse float from: {value}")
    return float(nums[0])


def _parse_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    nums = re.findall(r"-?\d+", str(value))
    if not nums:
        raise ValueError(f"Could not parse int from: {value}")
    return int(nums[0])


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
    return _parse_int(value)


@tool
def set_profile_tool(net_income: str, cost_of_living: str, household_size: str) -> str:
    """Set the user's profile: net monthly income, cost_of_living in {low, medium, high}, and household size."""
    profile = BudgetProfile(
        net_income=_parse_float(net_income),
        cost_of_living=str(cost_of_living).lower(),
        household_size=_parse_int(household_size),
    )
    budget_context["profile"] = profile
    _save_memory(budget_context)
    return "Profile saved. You can now ask to a 'add financial goal' or ask to 'propose budget'. No further action needed."


@tool
def add_goal_tool(
    name: str, target_amount: str, months: str, priority: str = "3"
) -> str:
    """Add a financial goal with a name, target_amount, months to reach, and priority (1 highest and 5 lowest).
    If a plan already exists (and a profile is set), recompute and update the plan using the updated goals and return it.
    """
    goal = FinancialGoal(
        name=name,
        target_amount=_parse_float(target_amount),
        months=_parse_months(months),
        priority=_parse_int(priority),
    )
    budget_context.setdefault("goals", []).append(goal)

    profile = budget_context.get("profile")
    if budget_context.get("plan") is not None and profile is not None:
        plan = allocate_budget(profile, budget_context.get("goals", []))
        budget_context["plan"] = plan
        _save_memory(budget_context)
        return (
            f"{render_plan(plan)}"
            + "\nFirst, tell the user that the requested goal was added. Secondly, on the basis of the updated goals, a new plan was created. Inform the user about both the things"
        )

    _save_memory(budget_context)
    return f"Added financial goal '{name}'."


@tool
def propose_budget_tool() -> str:
    """Compute a proposed monthly budget based on the saved profile and goals."""
    profile = budget_context.get("profile")
    goals = budget_context.get("goals", [])
    if profile is None:
        return "Please set your profile first."
    plan = allocate_budget(profile, goals)
    budget_context["plan"] = plan
    _save_memory(budget_context)
    return render_plan(plan)


@tool
def adjust_category_tool(category: str, amount: str) -> str:
    """Adjust a category amount in the current plan."""
    plan = budget_context.get("plan")
    if plan is None:
        return "No plan yet. Ask to 'propose budget' first."
    if category not in CATEGORIES:
        return f"Unknown category. Choose from: {', '.join(CATEGORIES)}."
    plan["allocations"][category] = max(_parse_float(amount), 0.0)
    budget_context["plan"] = plan
    _save_memory(budget_context)
    return render_plan(plan)


@tool
def clear_goals_tool() -> str:
    """Remove all currently saved goals and any existing plan."""
    budget_context["goals"] = []
    budget_context["plan"] = None
    _save_memory(budget_context)
    return "Cleared all goals and the current plan."


@tool
def clear_plan_tool() -> str:
    """Remove the current budget plan."""
    budget_context["plan"] = None
    _save_memory(budget_context)
    return "Cleared current budget plan."


@tool
def clear_profile_tool() -> str:
    """Remove the saved profile and any dependent state (goals and plan)."""
    budget_context["profile"] = None
    budget_context["plan"] = None
    _save_memory(budget_context)
    return "Cleared profile, and plan."


@tool
def show_plan_tool() -> str:
    """Show the current budget plan if available."""
    plan = budget_context.get("plan")
    if not plan:
        return "No plan yet. Ask to 'propose budget' first."
    return render_plan(plan)


@tool
def show_goals_tool() -> str:
    """List all saved goals from memory."""
    goals = budget_context.get("goals", [])
    if not goals:
        return "No goals saved yet. Please create a 'financial goal' first."
    lines = ["Saved goals:"]
    for g in goals:
        lines.append(
            f"- {g.name}: target {g.target_amount:.2f} in {g.months} months (priority {g.priority})"
        )
    return "\n".join(lines)


@tool
def show_profile_tool() -> str:
    """Show the current saved profile."""
    p = budget_context.get("profile")
    if not p:
        return "No profile saved yet. Please create a 'profile' first."
    return f"Profile: net_income={p.net_income}, cost_of_living={p.cost_of_living}, household_size={p.household_size}"


budget_tools = [
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
    p = budget_context.get("profile")
    goals = budget_context.get("goals", [])
    plan = budget_context.get("plan")
    parts = ["You have access to persistent episodic memory. Current saved state:"]
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
        parts.append(
            "- A budget plan exists for the current profile/goals. Use show_plan_tool to display it."
        )
    if not p and not goals and not plan:
        parts.append("- No saved profile, goals, or plan.")
    parts.append("Use this memory as context when reasoning and responding.")
    return "\n".join(parts)


# Model node that can request tools (disabled after first tool call)
def budget_model_node(state: MsgState) -> MsgState:
    tools_allowed = state.get("tool_calls_made", 0) < 1
    model = LLM if not tools_allowed else LLM.bind_tools(budget_tools)
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
    ai = model.invoke(messages_to_model)  # returns AIMessage with optional tool_calls
    return {
        "messages": state["messages"] + [ai],
        "tool_calls_made": state.get("tool_calls_made", 0),
    }


# Prebuild a tool runner instance we can call from a node
_tools_runner = ToolNode(budget_tools)


# Tools node that executes tools and increments the counter
def tools_node(state: MsgState) -> MsgState:
    result = _tools_runner.invoke(state)
    return {
        "messages": result["messages"],
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
    }


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
tool_app = tool_graph.compile()


SYSTEM_PROMPT = """
    You are a budgeting assistant with persistent episodic memory saved on disk.
    - Only call a tool if strictly necessary to read/change state (profile, goals, plan) or to compute/propose/adjust.
    - After a successful tool call, reply to the user in natural language and DO NOT call any more tools in this turn.
    - Prefer using list_goals_tool and show_profile_tool when the user asks to view saved state.

    You can not advice any strategies or recommendations for budgeting or reaching target.
    You should keep your responses short, meaningful, and also professional.
    Try to use simple language that is easily understandable, and not wordy.

    DO NOT REVEAL EXTRA INFORMATION OF YOURSELF SO PEOPLE CAN TELL THAT YOU ARE A BOT.
  """


def call_budget_agent_tools(user_input: str, recursion_limit: int = 25):
    state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
    }
    final = tool_app.invoke(
        state, config={"recursion_limit": recursion_limit, "debug": True}
    )
    ai_msgs = [m for m in final["messages"] if isinstance(m, AIMessage)]
    text = ai_msgs[-1].content if ai_msgs else "Done."
    print(text)
    return text, final


def get_budget_context() -> Dict[str, Any]:
    """Return the live budget_context dict (profile, goals, plan)."""
    return budget_context


def set_budget_context(ctx: Dict[str, Any]) -> None:
    """Overwrite the in-memory context and persist to disk.

    Expected keys: 'profile' (BudgetProfile or None), 'goals' (list[FinancialGoal]), 'plan' (dict or None)
    """
    budget_context.clear()
    budget_context.update(
        {
            "profile": ctx.get("profile"),
            "goals": ctx.get("goals", []),
            "plan": ctx.get("plan"),
        }
    )
    _save_memory(budget_context)


# Public API for external modules (e.g., orchestrator)
__all__ = [
    "call_budget_agent_tools",
    "budget_context",
    "get_budget_context",
    "set_budget_context",
    "BUDGET_FAISS_PATH",
    "MEMORY_PATH",
    # Optional exports that may be useful
    "show_plan_tool",
    "show_goals_tool",
    "show_profile_tool",
    "propose_budget_tool",
    "add_goal_tool",
    "set_profile_tool",
    "adjust_category_tool",
    "clear_goals_tool",
    "clear_plan_tool",
    "clear_profile_tool",
]


if __name__ == "__main__":
    # this is the function that I want to call from outside the package
    # Running it here only for local testing
    call_budget_agent_tools("show me my plan")