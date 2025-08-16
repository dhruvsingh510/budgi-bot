# Intelligent orchestrator with LangGraph routing
from resource_manager import resource_manager
from budget_service import BudgetService
from transaction_service import TransactionService
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ServiceRoute(BaseModel):
    """Pydantic model for service routing decision."""

    service: str = Field(
        description="The service to route to: 'budget' or 'transaction' 'general'"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the routing decision")


class OrchestratorState(TypedDict):
    """State for the orchestrator workflow."""

    user_input: str
    service_route: str
    response: str
    confidence: float
    reasoning: str


class BudgetBot:
    """Main application orchestrator with clean dependency injection."""

    def __init__(self):
        print("🚀 Initializing Intelligent Budget Bot with LangGraph routing...")

        # Initialize LLM for routing
        self.llm = resource_manager.llm

        # Initialize services with shared resources
        self.budget_service = BudgetService(
            llm=resource_manager.llm,
            text_splitter=resource_manager.text_splitter,
            embeddings=resource_manager.embeddings,
            vector_store=resource_manager.get_budget_vector_store(),
            memory_path=self._get_budget_memory_path(),
        )

        # Initialize transaction service
        self.transaction_service = TransactionService(
            llm=resource_manager.llm,
            text_splitter=resource_manager.text_splitter,
            embeddings=resource_manager.embeddings,
            vector_store=resource_manager.get_transaction_vector_store(),
            memory_path=self._get_transaction_memory_path(),
        )

        # Setup intelligent routing workflow
        self._setup_routing_workflow()

        print("✅ Intelligent Budget Bot with auto-routing initialized successfully!")

    def _get_budget_memory_path(self) -> str:
        """Get path to budget memory file."""
        return os.path.abspath(
            os.path.join(Path(__file__).parent, "..", "data", "budget_memory.json")
        )

    def _get_transaction_memory_path(self) -> str:
        """Get path to transaction memory file."""
        return os.path.abspath(
            os.path.join(Path(__file__).parent, "..", "data", "transaction_memory.json")
        )

    BOT_CAPABILITIES = """
      **BUDGET SERVICE capabilities:**
          - Set user profile (income, cost of living, household size)
          - Add/remove financial goals (savings targets, timelines, priorities)
          - Generate budget plans based on profile and goals
          - Adjust category allocations in budget plans
          - Show current budget plan, goals, and profile

          **TRANSACTION SERVICE capabilities:**
          - Add new transactions (with amount, category, item name)
          - Edit existing transactions
          - Delete transactions
          - Search transactions by category
          - View recent/similar transactions
          - Get all transactions grouped by category

          Transaction categories: Same as budget categories
    """

    BOT_EXAMPLES = """
      Examples:
          - "Set my income to $5000" → BUDGET
          - "Add goal to save $10000" → BUDGET  
          - "Show my budget plan" → BUDGET
          - "Add coffee $5" → TRANSACTION
          - "Show recent transactions" → TRANSACTION
          - "Edit my grocery purchase" → TRANSACTION
    """

    def _get_system_prompt(self) -> str:
        """Get the system prompt for intelligent routing."""
        return """
          You are an intelligent routing assistant for a personal finance bot. You need to determine whether a user's request should be handled by the BUDGET service, the TRANSACTION service, or the GENERAL response,.

          {BOT_CAPABILITIES}
          - Budget categories: Income, Housing & Utilities, Food & Groceries, Transportation, Insurance, Healthcare, Debt & Loans, Savings & Investments, Entertainment & Leisure, Shopping, Education, Personal Care, Travel & Holidays, Family & Childcare, Miscellaneous
          - Transaction categories: Same as budget categories

          **GENERAL response** - For greetings, introductions, and capability inquiries:
          - Basic greetings: "Hi", "Hello", "Hey", "Good morning"
          - Capability questions: "What can you do?", "What are your features?", "How do you work?"
          - Help requests: "Help me", "How do I get started?", "What should I do?"
          - Bot information: "Who are you?", "What are you?", "Tell me about yourself"

          **ROUTING GUIDELINES:**
          - Route to BUDGET for: profile setup, goal setting, budget planning, allocation adjustments, showing plans/goals/profiles
          - Route to TRANSACTION for: adding/editing/deleting expenses, viewing transaction history, searching past purchases
          - Keywords for BUDGET: "profile", "income", "goal", "plan", "budget", "allocate", "save", "target"
          - Keywords for TRANSACTION: "add", "buy", "bought", "spent", "purchase", "transaction", "history", "show recent", "edit", "delete"

          Use General response for the following prompts:
          - "Hi" → GENERAL (greeting)
          - "What can you do?" → GENERAL (capability inquiry)  
          - "How does this work?" → GENERAL (help request)
          - "Help me get started" → GENERAL (general help)

          {BOT_EXAMPLES}

          Analyze the user input and determine the most appropriate service.

          For simple inputs like "Hi" or "What can you do?", introduce yourself as BudgiBot, tell about your capabilities, and ask how you can help.
        """

    def _setup_routing_workflow(self):
        """Setup the LangGraph workflow for intelligent routing."""

        def route_node(state: OrchestratorState) -> OrchestratorState:
            """Determine which service to route to based on user input."""
            print("🧠 Analyzing user request...")

            prompt_template = PromptTemplate.from_template(
                """
                  {system_prompt}

                  User Input: "{user_input}"

                  Analyze this input and determine which service should handle it. Respond with valid JSON only:
                  {{
                    "service": "general" or "budget" or "transaction",
                    "confidence": 0.0 to 1.0,
                    "reasoning": "brief explanation for the choice"
                  }}
                """
            )

            prompt = prompt_template.format(
                system_prompt=self._get_system_prompt(), user_input=state["user_input"]
            )

            parser = PydanticOutputParser(pydantic_object=ServiceRoute)

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                parsed = parser.parse(response.content)

                state["service_route"] = parsed.service
                state["confidence"] = parsed.confidence
                state["reasoning"] = parsed.reasoning

                print(
                    f"🎯 Routing to {parsed.service.upper()} service (confidence: {parsed.confidence:.2f})"
                )
                print(f"💭 Reasoning: {parsed.reasoning}")

            except Exception as e:
                print(f"❌ Routing error: {e}")
                # Fallback: route to budget by default
                state["service_route"] = "budget"
                state["confidence"] = 0.5
                state["reasoning"] = "Fallback to budget service due to parsing error"

            return state

        def general_node(state: OrchestratorState) -> OrchestratorState:
            """Handle general greetings and capability inquiries with LLM."""
            print("🤖 Processing with General handler...")

            user_input = state["user_input"].lower()

            # Use LLM to generate contextual response for general queries
            general_prompt = f"""
              You are a helpful Personal Finance Bot. The user has sent a general greeting or capability inquiry: "{state['user_input']}"

              Respond in a friendly, helpful way. If it's a greeting, welcome them warmly. If they're asking about capabilities, explain what you can do clearly.

              Keep your response concise but informative. Include relevant examples of what they can ask you.

              Your capabilities:
              - Budget planning and goal setting
              - Expense tracking and transaction management  
              - Financial insights and recommendations
              - Natural language interaction

              Generate a helpful response:
              Avoid giving examples of what they can ask you.
              Avoid giving the same response repeatedly.

              Respond with a proper explanation on the bot's capbilities categorised by each bot:
              {self.BOT_CAPABILITIES}
              Make sure that each bot has it's own bullet point and it's capabilities are given as points nested in the bullet point.
              """

            try:
                response = self.llm.invoke([HumanMessage(content=general_prompt)])
                state["response"] = response.content
                print(general_prompt)
            except Exception as e:
                print(f"❌ Error in general handler: {e}")
                # Fallback response
                state[
                    "response"
                ] = """
                  👋 Hello! I'm your Personal Finance Bot. I can help you with budget planning and expense tracking. 

                  💰 Try: "Set my income to $5000" or "Add coffee $5"
                  ❓ Ask: "What can you do?" for more details

                  How can I help with your finances today?
                """

            return state

        def budget_node(state: OrchestratorState) -> OrchestratorState:
            """Process request using budget service."""
            print("💰 Processing with Budget service...")
            response, _ = self.budget_service.process_request(state["user_input"])
            state["response"] = response
            return state

        def transaction_node(state: OrchestratorState) -> OrchestratorState:
            """Process request using transaction service."""
            print("💳 Processing with Transaction service...")
            response, _ = self.transaction_service.process_request(state["user_input"])
            state["response"] = response
            return state

        def decide_service(state: OrchestratorState):
            """Route to the appropriate service based on routing decision."""
            service = state.get("service_route", "budget")
            if service == "transaction":
                return "transaction"
            elif service == "budget":
                return "budget"
            else:
                return "general"

        # Build the routing workflow
        self.routing_graph = StateGraph(OrchestratorState)
        self.routing_graph.add_node("route", route_node)
        self.routing_graph.add_node("budget", budget_node)
        self.routing_graph.add_node("transaction", transaction_node)
        self.routing_graph.add_node("general", general_node)

        # Add edges
        self.routing_graph.set_entry_point("route")
        self.routing_graph.add_conditional_edges("route", decide_service)
        self.routing_graph.add_edge("budget", END)
        self.routing_graph.add_edge("transaction", END)
        self.routing_graph.add_edge("general", END)

        # Compile the graph
        self.compiled_routing_graph = self.routing_graph.compile()

    def process_request(self, user_input: str) -> str:
        """Intelligently route request to appropriate service using LangGraph."""
        print(f"🔍 Processing: '{user_input}'")

        # Create initial state
        initial_state = OrchestratorState(
            user_input=user_input,
            service_route="",
            response="",
            confidence=0.0,
            reasoning="",
        )

        # Execute the routing workflow
        final_state = self.compiled_routing_graph.invoke(initial_state)

        # Store the routing state for API access
        self.last_routing_state = dict(final_state)

        return final_state.get("response", "Sorry, I couldn't process your request.")


def main():
    """Interactive loop with intelligent routing."""
    try:
        bot = BudgetBot()
        print("\n🤖 Intelligent Personal Finance Bot is ready! Type 'exit' to quit.")
        print(
            "🧠 AI will automatically route your requests to Budget or Transaction services"
        )
        print("📝 Just type naturally:")
        print("   • 'Set my income to $5000' (Budget)")
        print("   • 'Add coffee $5' (Transaction)")
        print("   • 'Show my budget plan' (Budget)")
        print("   • 'Show recent transactions' (Transaction)")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if user_input.lower() == "exit":
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    print("Please enter a valid input or 'exit' to quit.")
                    continue

                # Process with intelligent routing (no manual parsing needed!)
                response = bot.process_request(user_input)
                print(f"\n🤖 Bot: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to quit.")

    except Exception as e:
        print(f"❌ Failed to initialize Intelligent Bot: {e}")
        print("Please check your environment variables and dependencies.")


if __name__ == "__main__":
    main()
