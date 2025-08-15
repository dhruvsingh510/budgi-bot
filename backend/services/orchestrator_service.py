# Clean orchestrator with proper architecture
from resource_manager import resource_manager
from budget_service import BudgetService
from transaction_service import TransactionService
import os
from pathlib import Path


class BudgetBot:
    """Main application orchestrator with clean dependency injection."""

    def __init__(self):
        print("🚀 Initializing Budget Bot with clean architecture...")

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

        print("✅ Budget Bot with Transaction support initialized successfully!")

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

    def process_request(self, user_input: str, bot_type: str = "budget") -> str:
        """Route request to appropriate service."""
        if bot_type == "budget":
            response, _ = self.budget_service.process_request(user_input)
            return response
        elif bot_type == "transaction":
            response, _ = self.transaction_service.process_request(user_input)
            return response
        else:
            return "Unknown bot type. Use 'budget' or 'transaction'."


def main():
    """Interactive loop with clean architecture."""
    try:
        bot = BudgetBot()
        print("\n💰 Budget & Transaction Bot is ready! Type 'exit' to quit.")
        print("📝 Commands:")
        print(
            "   • budget: <message> - for budget questions (e.g., 'budget: show my plan')"
        )
        print(
            "   • transaction: <message> - for transactions (e.g., 'transaction: add coffee $5')"
        )
        print("   • <message> - defaults to budget mode")
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

                # Parse bot type and message
                if ":" in user_input:
                    bot_type, message = user_input.split(":", 1)
                    bot_type = bot_type.strip().lower()
                    message = message.strip()
                else:
                    bot_type = "budget"  # Default
                    message = user_input

                # Process the request
                response = bot.process_request(message, bot_type)
                print(f"\n🤖 Bot: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to quit.")

    except Exception as e:
        print(f"❌ Failed to initialize Budget Bot: {e}")
        print("Please check your environment variables and dependencies.")


if __name__ == "__main__":
    main()
