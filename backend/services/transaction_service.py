# Clean transaction service with dependency injection
import os
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from difflib import SequenceMatcher

from langgraph.graph import StateGraph, END
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class TransactionState(BaseModel):
    input: str = ""
    amount: Optional[float] = None
    category: Optional[str] = None
    item_name: Optional[str] = None
    action: Optional[str] = None
    result: Optional[dict] = None
    parse_attempts: int = 0
    needs_followup: bool = False
    followup_prompt: Optional[str] = None


class TransactionParse(BaseModel):
    action: str
    amount: Optional[float]
    category: Optional[str]
    item_name: Optional[str]


class TransactionService:
    """Clean transaction service with dependency injection."""

    # Constants
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

    # Timezone: IST (UTC+05:30)
    IST = timezone(timedelta(hours=5, minutes=30))

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

        # Load transactions from memory
        self.transactions = self._load_transactions()

        # State for handling follow-up interactions
        self.pending_add_context = None
        self.last_state_snapshot = None

        # Initialize the LangGraph workflow
        self._setup_workflow()

    def _load_transactions(self) -> List[Dict]:
        """Load transactions from persistent storage."""
        try:
            with open(self.memory_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("transactions", [])
            return data or []
        except Exception:
            return []

    def _save_transactions(self, transactions: List[Dict]) -> None:
        """Save transactions to persistent storage."""
        try:
            with open(self.memory_path, "w") as f:
                json.dump({"transactions": transactions}, f, indent=2)
        except Exception:
            pass

    # Transaction operations
    def add_transaction(
        self,
        input_text: str,
        amount: Optional[float] = None,
        category: Optional[str] = None,
        item_name: Optional[str] = None,
    ) -> Dict:
        """Add a new transaction to the transaction database."""

        # Create a persistent id for the vector store and transaction
        doc_id = str(uuid4())

        transaction = {
            "datetime": datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "category": category,
            "item_name": item_name,
            "input": input_text,
            "doc_id": doc_id,
        }

        # Update in-memory store
        self.transactions.append(transaction)
        self._save_transactions(self.transactions)

        # Add to vector store if available
        if self.vector_store and item_name:
            doc = Document(
                page_content=item_name,
                metadata={
                    "category": category or "Miscellaneous",
                    "amount": amount or 0.0,
                    "doc_id": doc_id,
                },
            )
            self.vector_store.add_documents([doc])

        return {
            "status": "success",
            "message": f"Added transaction: {item_name or 'Transaction'} for {amount or 'unspecified amount'}",
            "transaction": transaction,
        }

    def edit_transaction(
        self,
        input_text: str,
        amount: Optional[float] = None,
        category: Optional[str] = None,
        item_name: Optional[str] = None,
    ) -> Dict:
        """Edit an existing transaction based on best match."""

        # Find best matching transaction
        candidates = self._find_similar_transactions(
            input_text, amount, category, item_name
        )

        if not candidates:
            return {
                "status": "error",
                "message": "No matching transaction found to edit.",
                "transactions": [],
            }

        best_match = candidates[0]

        # Update fields if provided
        if amount is not None:
            best_match["amount"] = amount
        if category is not None:
            best_match["category"] = category
        if item_name is not None:
            best_match["item_name"] = item_name

        # Save updated transactions
        self._save_transactions(self.transactions)

        return {
            "status": "success",
            "message": f"Updated transaction: {best_match.get('item_name', 'Transaction')}",
            "transaction": best_match,
        }

    def delete_transaction(
        self,
        input_text: str,
        amount: Optional[float] = None,
        category: Optional[str] = None,
        item_name: Optional[str] = None,
    ) -> Dict:
        """Delete an existing transaction based on best match."""

        # Find best matching transaction
        candidates = self._find_similar_transactions(
            input_text, amount, category, item_name
        )

        if not candidates:
            return {
                "status": "error",
                "message": "No matching transaction found to delete.",
                "transactions": [],
            }

        best_match = candidates[0]

        # Remove from transactions list
        self.transactions = [
            t for t in self.transactions if t.get("doc_id") != best_match.get("doc_id")
        ]
        self._save_transactions(self.transactions)

        return {
            "status": "success",
            "message": f"Deleted transaction: {best_match.get('item_name', 'Transaction')}",
            "transaction": best_match,
        }

    def search_transaction_by_category(self, category: str) -> List[Dict]:
        """Search for transactions by category."""
        results = []
        for transaction in self.transactions:
            if transaction.get("category", "").lower() == category.lower():
                results.append(transaction)
        return results

    def get_recent_similar_transactions(
        self, input_text: str, k: int = 3
    ) -> List[Dict]:
        """Get recent transactions similar to the user input from the vector DB."""
        if not self.vector_store:
            return []

        try:
            similar_docs = self.vector_store.similarity_search(input_text, k=k)
            results = []

            for doc in similar_docs:
                doc_id = doc.metadata.get("doc_id")
                matching_transaction = next(
                    (t for t in self.transactions if t.get("doc_id") == doc_id), None
                )
                if matching_transaction:
                    results.append(matching_transaction)

            return results
        except Exception:
            return []

    def get_all_transactions_in_all_categories(self) -> Dict:
        """Return all transactions grouped by category with per-category totals and counts."""
        grouped = {}

        for transaction in self.transactions:
            category = transaction.get("category", "Miscellaneous")
            amount = transaction.get("amount", 0) or 0

            if category not in grouped:
                grouped[category] = {"transactions": [], "total": 0, "count": 0}

            grouped[category]["transactions"].append(transaction)
            grouped[category]["total"] += amount
            grouped[category]["count"] += 1

        return grouped

    # Helper methods for similarity matching
    def _find_similar_transactions(
        self,
        input_text: str,
        amount: Optional[float] = None,
        category: Optional[str] = None,
        item_name: Optional[str] = None,
    ) -> List[Dict]:
        """Find transactions similar to the given criteria."""

        candidates = []
        for transaction in self.transactions:
            score = self._score_candidate(
                input_text, amount, category, item_name, transaction
            )
            if score > 0.1:  # Minimum similarity threshold
                candidates.append((score, transaction))

        # Sort by score (descending) and return transactions
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [t for score, t in candidates]

    def _score_candidate(
        self,
        user_input: str,
        desired_amount: Optional[float],
        desired_category: Optional[str],
        desired_item: Optional[str],
        candidate: Dict,
    ) -> float:
        """Score how well a candidate transaction matches the criteria."""

        # Text similarity (item_name and input)
        text_score = 0.0
        if desired_item and candidate.get("item_name"):
            text_score = max(
                self._sequence_similarity(
                    desired_item.lower(), candidate["item_name"].lower()
                ),
                self._token_similarity(desired_item, candidate["item_name"]),
            )

        if candidate.get("input"):
            input_similarity = self._sequence_similarity(
                user_input.lower(), candidate["input"].lower()
            )
            text_score = max(text_score, input_similarity)

        # Amount similarity
        amount_score = self._amount_similarity(desired_amount, candidate.get("amount"))

        # Category similarity
        category_score = 0.0
        if desired_category and candidate.get("category"):
            if desired_category.lower() == candidate["category"].lower():
                category_score = 1.0
            else:
                category_score = self._sequence_similarity(
                    desired_category.lower(), candidate["category"].lower()
                )

        # Weighted combination
        weights = {"text": 0.5, "amount": 0.3, "category": 0.2}
        total_score = (
            weights["text"] * text_score
            + weights["amount"] * amount_score
            + weights["category"] * category_score
        )

        return total_score

    @staticmethod
    def _sequence_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
        tokens_b = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _amount_similarity(
        target: Optional[float], candidate: Optional[float]
    ) -> float:
        if target is None or candidate is None:
            return 0.0

        if target == 0 and candidate == 0:
            return 1.0

        diff = abs(target - candidate)
        max_val = max(abs(target), abs(candidate))
        return max(0.0, 1.0 - diff / max_val)

    def _setup_workflow(self):
        """Setup the LangGraph workflow for transaction processing."""

        def llm_parse_node(state: TransactionState) -> TransactionState:
            print("➡️ Running llm_parse_node")
            state.parse_attempts += 1

            # Fetch similar examples if vector store is available
            context = ""
            if self.vector_store:
                try:
                    similar_docs = self.vector_store.similarity_search(state.input, k=3)
                    context = "\n".join(
                        f"- {doc.page_content} ({doc.metadata.get('category')})"
                        for doc in similar_docs
                    )
                except Exception:
                    pass

            # Compose context-enhanced prompt
            prompt_template = PromptTemplate.from_template(
                """
                You are a transaction assistant. Use the examples below to help understand the user's input.

                Examples:
                {context}

                Now extract the following fields from the user input:
                - action: one of "add", "edit", "delete", "search_by_category", "get_recent", "get_all_by_category" (string)
                - amount: the transaction amount (float or null)
                - category: the transaction category mentioned below. If you are unsure of a category, set it to Miscellaneous (string)
                - item_name: the item name (string or null)

                Categories:
                {categories}

                User input: "{user_input}"

                Respond with valid JSON only:
                {{
                  "action": "...",
                  "amount": ...,
                  "category": "...",
                  "item_name": "..."
                }}
                """
            )

            prompt = prompt_template.format(
                context=context,
                categories="\n".join(f"- {cat}" for cat in self.CATEGORIES),
                user_input=state.input,
            )

            # Parse with LLM
            parser = PydanticOutputParser(pydantic_object=TransactionParse)

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                parsed = parser.parse(response.content)

                state.action = parsed.action
                state.amount = parsed.amount
                state.category = parsed.category
                state.item_name = parsed.item_name

            except Exception as e:
                print(f"❌ Parse error: {e}")
                state.action = "add"  # Default fallback

            return state

        def add_node(state: TransactionState) -> TransactionState:
            result = self.add_transaction(
                state.input, state.amount, state.category, state.item_name
            )
            state.result = result
            return state

        def edit_node(state: TransactionState) -> TransactionState:
            result = self.edit_transaction(
                state.input, state.amount, state.category, state.item_name
            )
            state.result = result
            return state

        def delete_node(state: TransactionState) -> TransactionState:
            result = self.delete_transaction(
                state.input, state.amount, state.category, state.item_name
            )
            state.result = result
            return state

        def search_node_by_category(state: TransactionState) -> TransactionState:
            result = self.search_transaction_by_category(state.category)
            state.result = {"transactions": result}
            return state

        def get_recent_node(state: TransactionState) -> TransactionState:
            result = self.get_recent_similar_transactions(state.input, 3)
            state.result = {"transactions": result}
            return state

        def get_all_by_category_node(state: TransactionState) -> TransactionState:
            result = self.get_all_transactions_in_all_categories()
            state.result = result
            return state

        def decide_next(state: TransactionState):
            print("🔄 Deciding next step...")

            action = state.action

            if action == "add":
                # Check if we need more info for add
                if not state.category or (not state.amount and not state.item_name):
                    state.needs_followup = True
                    return "followup"
                return "add"
            elif action == "edit":
                return "edit"
            elif action == "delete":
                return "delete"
            elif action == "search_by_category":
                return "search_by_category"
            elif action == "get_recent":
                return "get_recent"
            elif action == "get_all_by_category":
                return "get_all_by_category"
            else:
                return END

        # Build the workflow graph
        self.transaction_graph = StateGraph(TransactionState)
        self.transaction_graph.add_node("llm_parse", llm_parse_node)
        self.transaction_graph.add_node("add", add_node)
        self.transaction_graph.add_node("edit", edit_node)
        self.transaction_graph.add_node("delete", delete_node)
        self.transaction_graph.add_node("search_by_category", search_node_by_category)
        self.transaction_graph.add_node("get_recent", get_recent_node)
        self.transaction_graph.add_node("get_all_by_category", get_all_by_category_node)
        self.transaction_graph.add_node(
            "followup", lambda state: state
        )  # No-op for followup

        # Add edges
        self.transaction_graph.set_entry_point("llm_parse")
        self.transaction_graph.add_conditional_edges("llm_parse", decide_next)
        self.transaction_graph.add_edge("add", END)
        self.transaction_graph.add_edge("edit", END)
        self.transaction_graph.add_edge("delete", END)
        self.transaction_graph.add_edge("search_by_category", END)
        self.transaction_graph.add_edge("get_recent", END)
        self.transaction_graph.add_edge("get_all_by_category", END)
        self.transaction_graph.add_edge("followup", END)

        # Compile the graph
        self.compiled_graph = self.transaction_graph.compile()

    def _format_transaction(self, transaction: Dict) -> str:
        """Format a single transaction for display."""
        date = transaction.get("datetime", "Unknown date")
        amount = transaction.get("amount")
        category = transaction.get("category", "Uncategorized")
        item_name = transaction.get("item_name", "Unknown item")

        # Format amount
        amount_str = f"${amount:.2f}" if amount is not None else "No amount"

        # Format date (just the date part, not time)
        if date != "Unknown date":
            try:
                date_part = date.split(" ")[0]  # Get just the date part
            except:
                date_part = date
        else:
            date_part = date

        return f"  • {item_name} - {amount_str} ({category}) - {date_part}"

    def _format_transactions_list(self, transactions: List[Dict]) -> str:
        """Format a list of transactions for display."""
        if not transactions:
            return "No transactions to display."

        # Sort by date (most recent first)
        sorted_transactions = sorted(
            transactions, key=lambda x: x.get("datetime", ""), reverse=True
        )

        lines = []
        for transaction in sorted_transactions:
            lines.append(self._format_transaction(transaction))

        return "\n".join(lines)

    def _format_natural_language_response(self, final_state: Dict, result: Any) -> str:
        """Format the result into a natural language response."""
        action = final_state.get("action")
        category = final_state.get("category")
        amount = final_state.get("amount")
        item_name = final_state.get("item_name")

        if action == "add":
            if final_state.get("needs_followup"):
                missing = []
                if not amount:
                    missing.append("amount")
                if not item_name:
                    missing.append("item name")
                return f"Please provide the {' and '.join(missing)}. If you want to proceed without it, reply 'skip'."

            if result and result.get("status") == "success":
                return f"✅ {result.get('message', 'Transaction added successfully')}"
            else:
                return f"❌ Failed to add transaction: {result.get('message', 'Unknown error')}"

        elif action == "edit":
            if result and result.get("status") == "success":
                return f"✅ {result.get('message', 'Transaction updated successfully')}"
            else:
                return f"❌ {result.get('message', 'No matching transaction found to edit')}"

        elif action == "delete":
            if result and result.get("status") == "success":
                return f"✅ {result.get('message', 'Transaction deleted successfully')}"
            else:
                return f"❌ {result.get('message', 'No matching transaction found to delete')}"

        elif action == "search_by_category":
            transactions = result.get("transactions", []) if result else []
            if transactions:
                header = f"📋 Found {len(transactions)} transactions in {category} category:\n"
                transaction_list = self._format_transactions_list(transactions)
                return header + transaction_list
            else:
                return f"No transactions found in {category} category"

        elif action == "get_recent":
            transactions = result.get("transactions", []) if result else []
            if transactions:
                header = f"📋 Found {len(transactions)} similar recent transactions:\n"
                transaction_list = self._format_transactions_list(transactions)
                return header + transaction_list
            else:
                return "No similar transactions found"

        elif action == "get_all_by_category":
            if result:
                lines = ["📊 All Transactions by Category:\n"]
                total_transactions = 0

                # Sort categories by transaction count (descending)
                sorted_categories = sorted(
                    result.items(), key=lambda x: x[1].get("count", 0), reverse=True
                )

                for category_name, category_data in sorted_categories:
                    count = category_data.get("count", 0)
                    total = category_data.get("total", 0)
                    transactions = category_data.get("transactions", [])
                    total_transactions += count

                    lines.append(
                        f"💰 {category_name}: {count} transactions, Total: ${total:.2f}"
                    )

                    # Show all transactions for each category
                    if transactions:
                        all_transactions = sorted(
                            transactions,
                            key=lambda x: x.get("datetime", ""),
                            reverse=True,
                        )

                        for transaction in all_transactions:
                            lines.append(
                                f"    {self._format_transaction(transaction).strip()}"
                            )

                    lines.append("")  # Empty line between categories

                lines.append(
                    f"📈 Total: {total_transactions} transactions across {len(result)} categories"
                )
                return "\n".join(lines)
            else:
                return "No transactions found"

        return "I processed your request."

    def _redact_doc_ids(self, obj):
        """Remove sensitive information from the response."""
        SENSITIVE_KEYS = {"doc_id", "input"}
        if isinstance(obj, dict):
            return {
                k: self._redact_doc_ids(v)
                for k, v in obj.items()
                if k not in SENSITIVE_KEYS
            }
        elif isinstance(obj, list):
            return [self._redact_doc_ids(item) for item in obj]
        else:
            return obj

    def process_request(self, user_input: str) -> Tuple[str, Any]:
        """Process user request and return response."""
        print(f"🔍 User Input: {user_input}\n")

        # Handle follow-up interactions
        if self.pending_add_context:
            return self._handle_followup(user_input)

        # Normal flow
        state = TransactionState(input=user_input)
        final_state = self.compiled_graph.invoke(state)

        # Cache last state for robustness
        self.last_state_snapshot = final_state

        # If follow-up needed, save context and prompt user
        if final_state.get("needs_followup", False):
            self.pending_add_context = {
                "input": user_input,
                "amount": final_state.get("amount"),
                "category": final_state.get("category"),
                "item_name": final_state.get("item_name"),
            }
            redacted = self._redact_doc_ids(final_state.get("result"))
            message = self._format_natural_language_response(
                dict(final_state), redacted
            )
            return message, None

        # Normal completion
        redacted = self._redact_doc_ids(final_state.get("result"))
        message = self._format_natural_language_response(dict(final_state), redacted)
        return message, redacted

    def _handle_followup(self, user_input: str) -> Tuple[str, Any]:
        """Handle follow-up interactions for incomplete transactions."""
        follow_raw = user_input.strip()
        follow = follow_raw.lower()
        ctx = self.pending_add_context

        # If the reply indicates skipping missing fields
        if follow in {"skip", "no", "n", "none", "na"}:
            self.pending_add_context = None
            result = self.add_transaction(
                ctx.get("input", ""),
                ctx.get("amount"),
                ctx.get("category"),
                ctx.get("item_name"),
            )
            final_state = {
                "action": "add",
                "amount": ctx.get("amount"),
                "category": ctx.get("category"),
                "item_name": ctx.get("item_name"),
                "result": result,
            }
            redacted = self._redact_doc_ids(result)
            message = self._format_natural_language_response(final_state, redacted)
            return message, redacted

        # Try to parse the reply to fill in missing fields
        if ctx.get("amount") is None:
            # Extract numeric value if present
            digits = re.sub(r"[^0-9.]+", "", follow)
            if digits:
                try:
                    ctx["amount"] = float(digits)
                except Exception:
                    pass

        if ctx.get("item_name") is None and not follow in {
            "skip",
            "no",
            "n",
            "none",
            "na",
        }:
            # If user typed non-numeric text, treat as item_name
            if not re.fullmatch(r"[0-9.]+", follow):
                ctx["item_name"] = follow_raw

        # Proceed when we have at least category and one of amount/item_name
        if ctx.get("category") and (
            ctx.get("amount") is not None or ctx.get("item_name") is not None
        ):
            self.pending_add_context = None
            result = self.add_transaction(
                ctx.get("input", ""),
                ctx.get("amount"),
                ctx.get("category"),
                ctx.get("item_name"),
            )
            final_state = {
                "action": "add",
                "amount": ctx.get("amount"),
                "category": ctx.get("category"),
                "item_name": ctx.get("item_name"),
                "result": result,
            }
            redacted = self._redact_doc_ids(result)
            message = self._format_natural_language_response(final_state, redacted)
            return message, redacted

        # Still missing; ask again
        missing = []
        if ctx.get("amount") is None:
            missing.append("amount")
        if ctx.get("item_name") is None:
            missing.append("item name")
        prompt = f"Please provide the {' and '.join(missing)}. If you want to proceed without it, reply 'skip'."
        return prompt, None

    def get_transactions(self) -> List[Dict]:
        """Get all transactions."""
        return self.transactions

    def set_transactions(self, transactions: List[Dict]) -> None:
        """Set transactions list."""
        self.transactions = transactions
        self._save_transactions(transactions)
