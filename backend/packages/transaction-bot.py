# Importing all packages
from langgraph.graph import StateGraph, END
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from IPython.display import display, clear_output
from dotenv import load_dotenv
import ipywidgets as widgets
from uuid import uuid4
import os
import re
import json
from difflib import SequenceMatcher


# Loading env variables
load_dotenv()

# Timezone: IST (UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Global variables
TRANSACTIONS_MEMORY_PATH = os.path.join(os.getcwd(), "../data/transaction_memory.json")
TRANSACTION_FAISS_PATH = "../data/faiss_store_transaction"

# Load and save with locally saved transaction file 
def _load_transactions() -> list:
    try:
        with open(TRANSACTIONS_MEMORY_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("transactions", [])
        return data or []
    except Exception:
        return []

def _save_transactions(transactions: list) -> None:
    try:
        with open(TRANSACTIONS_MEMORY_PATH, "w") as f:
            json.dump({"transactions": transactions}, f, indent=2)
    except Exception:
        pass


#Global Variables
transaction_db = _load_transactions() # Initialize from disk
pending_add_context = None # Holds context when an 'add' intent is partially parsed and we need follow-up from user
last_state_snapshot = None # Snapshot of the last final_state to recover context if needed across turns


# Splitting of sample data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
# Setting up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load transaction FAISS store
db = FAISS.load_local(
    f"{TRANSACTION_FAISS_PATH}",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Loaded existing transactions FAISS")

# Initialize LLM
LLM = ChatGroq(
    model_name=os.environ.get("LITELLM_MODEL"),
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# API Tools that will be used by the LLM
def add_transaction(
    input: str,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    item_name: Optional[str] = None
) -> dict:
    """Add a new transaction to the transaction_db. Amount, category and item_name may be omitted."""
    
    # Create a persistent id for the vector store and transaction
    doc_id = str(uuid4())

    transaction = {
        "datetime": datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
        "amount": amount,
        "category": category,
        "item_name": item_name,
        "input": input,
        "doc_id": doc_id,
    }
    
    # Save to in-memory DB
    transaction_db.append(transaction)

    # Save to vector DB
    doc = Document(
        page_content=input,
        metadata={
            "doc_id": doc_id,
            "amount": amount,
            "category": category,
            "item_name": item_name,
            "datetime": transaction["datetime"],
            "source": "user"
        }
    )
    db.add_documents([doc], ids=[doc_id])
    db.save_local(TRANSACTION_FAISS_PATH)

    _save_transactions(transaction_db)
    print("✅ Stored transaction in FAISS vector DB")
    return transaction


def edit_transaction(
    input: str,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    item_name: Optional[str] = None
) -> dict:
    """
    Edit a transaction by searching for the most similar past transaction using the user input.
    Can update amount, category, and/or item_name if provided.
    """
    # Retrieve similar transactions from vector DB
    similar_docs = db.similarity_search(input, k=1)
    if not similar_docs:
        return {"error": "No matching transaction found to edit."}
    
    best_match = similar_docs[0]
    metadata = best_match.metadata
    original_input = best_match.page_content

    # Locate and update the transaction in memory
    for i, t in enumerate(transaction_db):
        # Prefer matching by doc_id when available; fall back to input+datetime
        same_doc = (t.get("doc_id") and t.get("doc_id") == metadata.get("doc_id"))
        same_input_dt = (
            t.get("input") == original_input and t.get("datetime") == metadata.get("datetime")
        )
        if same_doc or same_input_dt:
            updated_transaction = t.copy()
            
            # Update fields if new values are provided
            if amount is not None:
                updated_transaction["amount"] = amount
            if category is not None:
                updated_transaction["category"] = category
            if item_name is not None:
                updated_transaction["item_name"] = item_name

            # Update both in-memory DB and vector DB
            transaction_db[i] = updated_transaction
            
            # Delete old doc from FAISS using doc_id (if available)
            doc_id = metadata.get("doc_id")
            if doc_id:
                try:
                    db.delete([doc_id])
                except ValueError:
                    # If id not found, fall back to similarity removal of original_input
                    try:
                        # Find a close match and delete its id if present
                        candidates = db.similarity_search(original_input, k=3)
                        for c in candidates:
                            cid = c.metadata.get("doc_id")
                            if cid:
                                try:
                                    db.delete([cid])
                                    break
                                except Exception:
                                    continue
                    except Exception:
                        pass
            else:
                print("⚠️ Warning: doc_id missing; skipping delete")

            # Add updated doc with same doc_id so we replace the vector
            new_doc_id = metadata.get("doc_id") or str(uuid4())
            new_doc = Document(
                page_content=input,
                metadata={
                    "doc_id": new_doc_id,
                    "datetime": updated_transaction["datetime"],
                    "amount": updated_transaction["amount"],
                    "category": updated_transaction["category"],
                    "item_name": updated_transaction["item_name"],
                    "input": input,
                    "source": "user"
                }
            )
            db.add_documents([new_doc], ids=[new_doc_id])
            db.save_local(TRANSACTION_FAISS_PATH)  # Persist changes

            _save_transactions(transaction_db)
            return {
                "updated_transaction": updated_transaction,
                "matched_on": original_input
            }

    return {"error": "Transaction found in vector DB but not in in-memory DB."}


def search_transaction_by_category(category: str) -> List[dict]:
    """Search for transactions by category."""
    results = []
    for t in transaction_db:
        if (t['category'] == category):
            results.append(t)
    return results


def get_recent_similar_transactions(input: str, k: int = 3) -> List[dict]:
    """
    Get recent transactions similar to the user input from the vector DB.
    Returns the top-k most recent similar transactions.
    """
    similar_docs = db.similarity_search(input, k=10, filter={"source": "user"})  # get more for filtering

    if not similar_docs:
        return []

    # Filter and sort by datetime (descending)
    parsed_docs = []
    for doc in similar_docs:
        metadata = doc.metadata
        dt_str = metadata.get("datetime")
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=IST)
        except Exception:
            dt = datetime.min.replace(tzinfo=IST)  # fallback for malformed datetimes

        parsed_docs.append({
            "input": doc.page_content,
            "amount": metadata.get("amount"),
            "category": metadata.get("category"),
            "item_name": metadata.get("item_name"),
            "datetime": dt_str,
            "datetime_obj": dt
        })

    # Sort by most recent
    parsed_docs.sort(key=lambda x: x["datetime_obj"], reverse=True)

    # Return top-k without the datetime_obj helper
    return [{k: v for k, v in doc.items() if k != "datetime_obj"} for doc in parsed_docs[:k]]

def get_all_transactions_in_all_categories() -> dict:
    """Return all transactions grouped by category with per-category totals and counts."""
    grouped = {}
    for t in transaction_db:
        cat = t.get("category") or "Uncategorized"
        grouped.setdefault(cat, []).append(t)

    summary = {}
    for cat, txs in grouped.items():
        total = sum(
            (a for a in (t.get("amount") for t in txs) if isinstance(a, (int, float))),
            0.0,
        )
        txs_sorted = sorted(txs, key=lambda x: x.get("datetime") or "", reverse=True)
        summary[cat] = {
            "count": len(txs_sorted),
            "total_amount": round(total, 2),
            "transactions": txs_sorted,
        }

    grand_total = sum(
        (a for a in (t.get("amount") for t in transaction_db) if isinstance(a, (int, float))),
        0.0,
    )
    return {
        "grand_count": len(transaction_db),
        "grand_total_amount": round(grand_total, 2),
        "categories": summary,
    }


# Fuzzy-matching edit_transaction overriding previous definition

def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _amount_similarity(target: Optional[float], candidate: Optional[float]) -> float:
    if target is None or candidate is None:
        return 0.0
    diff = abs(float(target) - float(candidate))
    if target == 0:
        return 1.0 if diff == 0 else 0.0
    # Normalize by target; clamp to [0,1]
    rel = min(diff / max(abs(target), 1e-6), 1.0)
    return max(0.0, 1.0 - rel)


def _score_candidate(
    user_input: str,
    desired_amount: Optional[float],
    desired_category: Optional[str],
    desired_item_name: Optional[str],
    transaction: dict
) -> float:
    # Weights
    WEIGHT_CATEGORY_EXACT = 3.0
    WEIGHT_CATEGORY_PARTIAL = 0.75
    WEIGHT_ITEM_SIMILARITY = 2.0
    WEIGHT_AMOUNT = 2.0
    WEIGHT_INPUT_TOKEN_OVERLAP = 1.5

    # Category score
    score = 0.0
    t_cat = _normalize_text(transaction.get("category"))
    d_cat = _normalize_text(desired_category)
    if d_cat:
        if t_cat == d_cat:
            score += WEIGHT_CATEGORY_EXACT
        else:
            # partial overlap on tokens
            overlap = len(_tokenize(t_cat) & _tokenize(d_cat))
            if overlap > 0:
                score += WEIGHT_CATEGORY_PARTIAL

    # Item similarity
    t_item = _normalize_text(transaction.get("item_name"))
    d_item = _normalize_text(desired_item_name)
    if d_item and t_item:
        score += WEIGHT_ITEM_SIMILARITY * _sequence_similarity(d_item, t_item)

    # Amount closeness
    t_amt = transaction.get("amount")
    score += WEIGHT_AMOUNT * _amount_similarity(desired_amount, t_amt)

    # Input token overlap
    t_input = _normalize_text(transaction.get("input"))
    tokens_user = _tokenize(_normalize_text(user_input))
    tokens_tx = _tokenize(t_input + " " + t_item + " " + t_cat)
    overlap = len(tokens_user & tokens_tx)
    if tokens_user:
        score += WEIGHT_INPUT_TOKEN_OVERLAP * (overlap / max(len(tokens_user), 1))

    return score


def _parse_ist(dt_str: Optional[str]):
    if not dt_str:
        return datetime.min.replace(tzinfo=IST)
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=IST)
    except Exception:
        return datetime.min.replace(tzinfo=IST)


def edit_transaction(
    input: str,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    item_name: Optional[str] = None
) -> dict:
    """Edit transaction using robust fuzzy matching over transaction_db.
        Scoring considers:
        - exact/partial category match
        - item name string similarity
        - amount closeness
        - token overlap with prior input/category/item

        Tie-breaks prefer the most recent transaction.
    """
    if not transaction_db:
        return {"error": "No transactions found to edit."}

    # Score all candidates
    scored: list[tuple[float, int, dict]] = []
    for idx, tx in enumerate(transaction_db):
        s = _score_candidate(input, amount, category, item_name, tx)
        if s > 0.0:
            scored.append((s, idx, tx))

    if not scored:
        return {"error": "No matching transaction found to edit."}

    # Sort by score desc, then by recency desc
    scored.sort(key=lambda x: (x[0], _parse_ist(x[2].get("datetime"))), reverse=True)
    _, best_index, best_tx = scored[0]

    # Build updated transaction (preserve doc_id and datetime)
    updated_tx = best_tx.copy()
    if amount is not None:
        updated_tx["amount"] = amount
    if category is not None:
        updated_tx["category"] = category
    if item_name is not None:
        updated_tx["item_name"] = item_name

    transaction_db[best_index] = updated_tx

    # Sync vector store: replace document with same doc_id when present
    doc_id = best_tx.get("doc_id")
    if doc_id:
        try:
            db.delete([doc_id])
        except Exception:
            pass
        try:
            new_doc = Document(
                page_content=input,
                metadata={
                    "doc_id": doc_id,
                    "datetime": updated_tx.get("datetime"),
                    "amount": updated_tx.get("amount"),
                    "category": updated_tx.get("category"),
                    "item_name": updated_tx.get("item_name"),
                    "input": input,
                    "source": "user",
                },
            )
            db.add_documents([new_doc], ids=[doc_id])
            db.save_local(TRANSACTION_FAISS_PATH)
        except Exception:
            pass

    _save_transactions(transaction_db)
    return {
        "updated_transaction": updated_tx,
        "matched_on": best_tx.get("item_name") or best_tx.get("input") or "transaction",
    }


def delete_transaction(
    input: str,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    item_name: Optional[str] = None
) -> dict:
    """Delete a transaction using the same fuzzy matching strategy as edit_transaction.
    Considers category, item name similarity, amount closeness, and input token overlap;
    tie-breaks by most recent. Removes from in-memory DB and vector store.
    """
    if not transaction_db:
        return {"error": "No transactions found to delete."}

    # Score all candidates
    scored: list[tuple[float, int, dict]] = []
    for idx, tx in enumerate(transaction_db):
        s = _score_candidate(input, amount, category, item_name, tx)
        if s > 0.0:
            scored.append((s, idx, tx))

    if not scored:
        return {"error": "No matching transaction found to delete."}

    # Sort by score desc, then by recency desc
    scored.sort(key=lambda x: (x[0], _parse_ist(x[2].get("datetime"))), reverse=True)
    _, best_index, best_tx = scored[0]

    # Delete from vector store by doc_id if available
    doc_id = best_tx.get("doc_id")
    if doc_id:
        try:
            db.delete([doc_id])
        except Exception:
            pass

    # Remove from in-memory DB
    deleted_tx = transaction_db.pop(best_index)

    # Persist changes
    try:
        db.save_local(TRANSACTION_FAISS_PATH)
    except Exception:
        pass

    _save_transactions(transaction_db)

    # Build redacted payload for response (omit doc_id)
    redacted_deleted = {k: v for k, v in deleted_tx.items() if k != "doc_id"}

    return {
        "deleted_transaction": redacted_deleted,
        "matched_on": best_tx.get("item_name") or best_tx.get("input") or "transaction",
    }


# State Object that is passed in Graph

# from requests import get


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

# Output Parser 
class TransactionParse(BaseModel):
    action: str
    amount: Optional[float]
    category: Optional[str]
    item_name: Optional[str]

def retrieve_similar_examples(query: str, k: int = 3) -> List[Document]:
    return db.similarity_search(query, k=k)

# Node function of Graph
def llm_parse_node(state: TransactionState) -> TransactionState:
    print("➡️ Running llm_parse_node")
    state.parse_attempts += 1

    # Fetch similar examples
    similar_docs = retrieve_similar_examples(state.input)
    context = "\n".join(
        f"- {doc.page_content} ({doc.metadata.get('category')})"
        for doc in similar_docs
    )

    print("🧠 Similar examples:\n", context)

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
        - Income
        - Housing & Utilities
        - Food & Groceries
        - Transportation
        - Insurance
        - Healthcare
        - Debt & Loans
        - Savings & Investments
        - Entertainment & Leisure
        - Shopping
        - Education
        - Personal Care
        - Travel & Holidays
        - Family & Childcare
        - Miscellaneous

        If a field is missing, set it to null.

        If a user is trying to change the category for a transaction, make sure that the new transaction is present in the Categories mentioned above. If not, then set the category to Miscellaneous.

        DO NOT REVEAL EXTRA INFORMATION OF YOURSELF SO PEOPLE CAN TELL THAT YOU ARE A BOT.

        User input: {input}

        {format_instructions}
        """
    )

    parser = PydanticOutputParser(pydantic_object=TransactionParse)
    format_instructions = parser.get_format_instructions()
    chain = prompt_template | LLM | parser
    parsed: TransactionParse = chain.invoke({"input": state.input, "context": context, "format_instructions": format_instructions})
    

    print("🧠 Parsed Output:", parsed)

    state.action = parsed.action
    state.amount = parsed.amount
    state.category = parsed.category
    state.item_name = parsed.item_name

    return state

def add_node(state: TransactionState) -> TransactionState:
    result = add_transaction(state.input, state.amount, state.category, state.item_name)
    state.result = result
    return state


def edit_node(state: TransactionState) -> TransactionState:
    result = edit_transaction(state.input, state.amount, state.category, state.item_name)
    state.result = result
    return state

def delete_node(state: TransactionState) -> TransactionState:
    result = delete_transaction(state.input, state.amount, state.category, state.item_name)
    state.result = result
    return state

def search_node_by_category(state: TransactionState) -> TransactionState:
    result = search_transaction_by_category(state.category)
    state.result = result
    return state

def get_recent_node(state: TransactionState) -> TransactionState:
    result = get_recent_similar_transactions(state.input, 1)
    state.result = result
    return state

def get_all_by_category_node(state: TransactionState) -> TransactionState:
    result = get_all_transactions_in_all_categories()
    state.result = result
    return state

# Decision Function
def decide_next(state: TransactionState):
    print("🔄 Deciding next step...")
    print("State:", state)

    max_attempts = 1
    REPARSE_LIMIT = 3  # maximum allowed re-parses of llm_parse

    # 1) Add flow: if missing required fields after attempts, ask user for follow-up
    if state.action == "add" and (state.amount is None or state.item_name is None):
        if state.parse_attempts >= max_attempts:
            missing = []
            if state.amount is None:
                missing.append("amount")
            if state.item_name is None:
                missing.append("item name")
            missing_str = " and ".join(missing)
            state.needs_followup = True
            state.followup_prompt = (
                f"I can add this under {state.category or 'unspecified category'}. "
                f"Please provide the {missing_str}. If you want to proceed without it, reply 'skip'."
            )
            print("🟡 Asking user for follow-up:", state.followup_prompt)
            return END

    # 2) Re-enter parse if still missing fields for add/edit, but stop after REPARSE_LIMIT
    if state.action == "add" and (state.amount is None or state.category is None or state.item_name is None):
        if (state.parse_attempts - 1) >= REPARSE_LIMIT:
            state.result = {"error": "recursion limit reached in the parsing step"}
            return END
        print("🔁 Re-entering llm_parse (missing add fields)")
        return "llm_parse"
    if state.action == "edit" and (state.category is None or state.item_name is None):
        if (state.parse_attempts - 1) >= REPARSE_LIMIT:
            state.result = {"error": "recursion limit reached in the parsing step"}
            return END
        print("🔁 Re-entering llm_parse (missing edit fields)")
        return "llm_parse"

    # 3) Route to concrete actions as soon as recognized
    if state.action == "add":
        print("✅ Going to add")
        return "add"
    if state.action == "edit":
        print("✅ Going to edit")
        return "edit"
    if state.action == "search_by_category":
        print("✅ Going to search by category")
        return "search_by_category"
    if state.action == "get_recent":
        print("✅ Going to get_recent")
        return "get_recent"
    if state.action == "get_all_by_category":
        print("✅ Going to get_all_by_category")
        return "get_all_by_category"
    if state.action == "delete":
        print("✅ Going to delete")
        return "delete"

    # 4) For unknown intents, only then apply the generic attempts guard
    if state.parse_attempts >= max_attempts:
        state.result = {
            "error": "❌ Unable to understand input after multiple attempts. Please rephrase."
        }
        return END

    print("⏹ Ending graph")
    return END

# Build the Graph
def build_transaction_graph():
    graph = StateGraph(TransactionState)
    graph.add_node("llm_parse", llm_parse_node)
    graph.add_node("add", add_node)
    graph.add_node("edit", edit_node)
    graph.add_node("delete", delete_node)
    graph.add_node("search_by_category", search_node_by_category)
    graph.add_node("get_recent", get_recent_node)
    graph.add_node("get_all_by_category", get_all_by_category_node)
    graph.add_edge("get_all_by_category", END)
    graph.add_edge("get_recent", END)
    graph.add_edge("add", END)
    graph.add_edge("edit", END)
    graph.add_edge("delete", END)
    graph.add_edge("search_by_category", END)
    graph.add_conditional_edges("llm_parse", decide_next)
    graph.set_entry_point("llm_parse")
    return graph.compile()

transaction_graph = build_transaction_graph()

# Example to call

def format_natural_language_response(final_state, result) -> str:
    action = final_state.get("action")
    category = final_state.get("category")
    item_name = final_state.get("item_name")
    amount = final_state.get("amount")

    # If a follow-up is needed, surface the prompt
    if final_state.get("needs_followup") and final_state.get("followup_prompt"):
        return final_state.get("followup_prompt")

    if isinstance(result, dict) and "error" in result:
        return f"{result['error']}"

    if action == "add" and isinstance(result, dict):
        amt_text = f"{amount}" if amount is not None else "an unspecified amount"
        item_text = f" for {item_name}" if item_name else ""
        cat_text = f" in category {category}" if category else ""
        return f"Added a transaction of {amt_text}{cat_text}{item_text}."

    if action == "edit" and isinstance(result, dict):
        updated = result.get("updated_transaction", {})
        amt_text = f"{updated.get('amount')}" if updated.get("amount") is not None else "an unspecified amount"
        item_text = f" for {updated.get('item_name')}" if updated.get("item_name") else ""
        cat_text = f" in category {updated.get('category')}" if updated.get("category") else ""
        return f"Updated the transaction to {amt_text}{cat_text}{item_text}."

    if action == "delete" and isinstance(result, dict):
        deleted = result.get("deleted_transaction", {})
        amt_text = f"{deleted.get('amount')}" if deleted.get("amount") is not None else "an unspecified amount"
        item_text = f" for {deleted.get('item_name')}" if deleted.get("item_name") else ""
        cat_text = f" in category {deleted.get('category')}" if deleted.get("category") else ""
        return f"Deleted the transaction of {amt_text}{cat_text}{item_text}."

    if action == "search_by_category" and isinstance(result, list):
        n = len(result)
        if n == 0:
            return f"No transactions found in category {category}."
        txs_sorted = sorted(result, key=lambda x: x.get("datetime") or "", reverse=True)
        lines = [f"Transactions in category {category} — {n} total:"]
        for t in txs_sorted:
            amt = t.get("amount")
            name = t.get("item_name") or "unspecified item"
            dt = t.get("datetime")
            amt_text = f"{amt:.2f}" if isinstance(amt, (int, float)) else "unspecified amount"
            lines.append(f"- {dt}: {name} ({amt_text})")
        return "\n".join(lines)

    if action == "get_recent" and isinstance(result, list):
        if not result:
            return "No similar recent transactions found."
        lines = ["Most recent similar transaction(s):"]
        for t in result[:3]:
            amt = t.get("amount")
            name = t.get("item_name") or "unspecified item"
            cat = t.get("category")
            dt = t.get("datetime")
            amt_text = f"{amt}" if amt is not None else "unspecified amount"
            cat_text = f" in {cat}" if cat else ""
            lines.append(f"- {dt}: {name}{cat_text} ({amt_text})")
        return "\n".join(lines)

    if action == "get_all_by_category" and isinstance(result, dict):
        cats = result.get("categories", {})
        total_count = result.get("grand_count", 0)
        grand_total = result.get("grand_total_amount", 0.0)
        if total_count == 0:
            return "No transactions recorded yet."
        lines = [f"All transactions grouped by category — {total_count} total, sum {grand_total:.2f}:"]
        for cat in sorted(cats.keys()):
            info = cats[cat]
            lines.append(f"- {cat}: {info['count']} tx, total {info['total_amount']:.2f}")
            for t in info["transactions"]:
                amt = t.get("amount")
                name = t.get("item_name") or "unspecified item"
                dt = t.get("datetime")
                amt_text = f"{amt:.2f}" if isinstance(amt, (int, float)) else "unspecified amount"
                lines.append(f"  - {dt}: {name} ({amt_text})")
        return "\n".join(lines)

    # Fallbacks
    if isinstance(result, dict):
        parts = []
        for k, v in result.items():
            parts.append(f"{k}: {v}")
        return "Result:\n" + "\n".join(parts)

    return str(result)


def _redact_doc_ids(obj):
    SENSITIVE_KEYS = {"doc_id", "input"}
    if isinstance(obj, dict):
        return {k: _redact_doc_ids(v) for k, v in obj.items() if k not in SENSITIVE_KEYS}
    if isinstance(obj, list):
        return [_redact_doc_ids(v) for v in obj]
    return obj


def call_transaction_agent(user_input: str):
    global pending_add_context
    print(f"🔍 User Input: {user_input}\n")

    # If we are waiting for follow-up (amount/item) and the user responded
    if pending_add_context:
        follow_raw = user_input.strip()
        follow = follow_raw.lower()
        ctx = pending_add_context

        # If the reply indicates skipping missing fields
        if follow in {"skip", "no", "n", "none", "na"}:
            pending_add_context = None
            result = add_transaction(ctx.get("amount"), ctx.get("category"), ctx.get("item_name"), ctx.get("input"))
            final_state = {
                "action": "add",
                "amount": ctx.get("amount"),
                "category": ctx.get("category"),
                "item_name": ctx.get("item_name"),
                "result": result
            }
            redacted = _redact_doc_ids(result)
            message = format_natural_language_response(final_state, redacted)
            print(message)
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

        if ctx.get("item_name") is None and ctx.get("amount") is not None and not follow in {"skip", "no", "n", "none", "na"}:
            # If amount is now present and item is still missing, and user typed text, treat as item_name
            if not re.fullmatch(r"[0-9.]+", follow):
                ctx["item_name"] = follow_raw

        # If we still miss item_name and user typed non-numeric text first
        if ctx.get("item_name") is None and not re.fullmatch(r"[0-9.]+", follow) and follow not in {"skip", "no", "n", "none", "na"}:
            ctx["item_name"] = follow_raw

        # Proceed when we have at least category and one of amount/item_name
        if ctx.get("category") and (ctx.get("amount") is not None or ctx.get("item_name") is not None):
            pending_add_context = None
            result = add_transaction(ctx.get("amount"), ctx.get("category"), ctx.get("item_name"), ctx.get("input"))
            final_state = {
                "action": "add",
                "amount": ctx.get("amount"),
                "category": ctx.get("category"),
                "item_name": ctx.get("item_name"),
                "result": result
            }
            redacted = _redact_doc_ids(result)
            message = format_natural_language_response(final_state, redacted)
            print(message)
            return message, redacted

        # Still missing; ask again
        missing = []
        if ctx.get("amount") is None:
            missing.append("amount")
        if ctx.get("item_name") is None:
            missing.append("item name")
        prompt = f"Please provide the {' and '.join(missing)}. If you want to proceed without it, reply 'skip'."
        print(prompt)
        return prompt, None

    # Normal flow
    state = TransactionState(input=user_input)
    final_state = transaction_graph.invoke(state)

    # Cache last_state for robustness
    globals()["last_state_snapshot"] = final_state

    # If follow-up needed, save context and prompt user (primary path)
    if final_state.get("needs_followup"):
        pending_add_context = {
            "input": user_input,
            "amount": final_state.get("amount"),
            "category": final_state.get("category"),
            "item_name": final_state.get("item_name")
        }
        redacted = _redact_doc_ids(final_state.get("result"))
        message = format_natural_language_response(final_state, redacted)
        print(message)
        return message, None

    # Fallback: if graph ended without result and we can infer missing add fields, prompt and cache context
    if (
        final_state.get("result") is None
        and final_state.get("action") == "add"
        and (final_state.get("amount") is None or final_state.get("item_name") is None)
    ):
        pending_add_context = {
            "input": user_input,
            "amount": final_state.get("amount"),
            "category": final_state.get("category"),
            "item_name": final_state.get("item_name")
        }
        missing = []
        if final_state.get("amount") is None:
            missing.append("amount")
        if final_state.get("item_name") is None:
            missing.append("item name")
        missing_str = " and ".join(missing)
        prompt = (
            f"I can add this under {final_state.get('category') or 'unspecified category'}. "
            f"Please provide the {missing_str}. If you want to proceed without it, reply 'skip'."
        )
        print(prompt)
        return prompt, None

    result = final_state.get("result")
    redacted = _redact_doc_ids(result)
    message = format_natural_language_response(final_state, redacted)
    print(message)

    return message, redacted

# this is the function that I want to call from outside the package
call_transaction_agent("edit transaction in shopping cateogry by setting amount to 12")