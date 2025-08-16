from codecs import ignore_errors
import os
import shutil
from uuid import uuid4
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


TRANSACTION_FAISS_PATH = "../data/faiss_store_transaction"
BUDGET_FAISS_PATH = "../data/faiss_store_budget"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_folder_size(path):
    return sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(path)
        for filename in filenames
    )


transaction_examples = [
    {
        "input": "Paid rent for August",
        "amount": None,
        "item_name": "rent",
        "category": "Rent",
    },
    {
        "input": "Gave landlord 12,000 for rent",
        "amount": 12000,
        "item_name": "rent",
        "category": "Rent",
    },
    {
        "input": "Netflix charged my card",
        "amount": None,
        "item_name": "Netflix",
        "category": "Subscriptions",
    },
    {
        "input": "Monthly subscription to Netflix",
        "amount": None,
        "item_name": "Netflix",
        "category": "Subscriptions",
    },
    {
        "input": "Bought veggies and snacks from DMart",
        "amount": None,
        "item_name": "DMart",
        "category": "Groceries",
    },
    {
        "input": "Shopping at Walmart – groceries and drinks",
        "amount": None,
        "item_name": "Walmart",
        "category": "Groceries",
    },
    {
        "input": "Recharge for electricity board",
        "amount": None,
        "item_name": "electricity board",
        "category": "Utilities",
    },
    {
        "input": "TNEB current bill paid",
        "amount": None,
        "item_name": "TNEB",
        "category": "Utilities",
    },
    {
        "input": "Watched Barbie movie at INOX",
        "amount": None,
        "item_name": "INOX",
        "category": "Shopping & Entertainment",
    },
    {
        "input": "Cinema with friends at PVR",
        "amount": None,
        "item_name": "PVR",
        "category": "Shopping & Entertainment",
    },
    {
        "input": "Renewed Tata AIG insurance for car",
        "amount": None,
        "item_name": "Tata AIG",
        "category": "Insurance",
    },
    {
        "input": "Paid ICICI car insurance",
        "amount": None,
        "item_name": "ICICI",
        "category": "Insurance",
    },
    {
        "input": "Consultation at Apollo Hospital",
        "amount": None,
        "item_name": "Apollo Hospital",
        "category": "Health",
    },
    {
        "input": "Doctor visit charges",
        "amount": None,
        "item_name": "Doctor",
        "category": "Health",
    },
    {
        "input": "Recharged metro card",
        "amount": None,
        "item_name": "metro card",
        "category": "Transport",
    },
    {
        "input": "Added ₹200 to metro pass",
        "amount": 200,
        "item_name": "metro pass",
        "category": "Transport",
    },
]

budget_guideline_texts = [
    {
        "title": "Single Professional - High COL",
        "content": (
            "Archetype: single professional in high cost-of-living city.\n"
            "Baseline monthly allocation as percent of net income:\n"
            "- Rent: 30-40%\n- Utilities: 5-8%\n- Groceries: 8-12%\n- Transport: 5-10%\n"
            "- Health: 3-5%\n- Subscriptions: 2-5%\n- Shopping & Entertainment: 5-10%\n"
            "Elasticity: Cut order -> Shopping & Entertainment (high), Subscriptions (med), Transport/Groceries (med), Health (low).\n"
            "Floors: Groceries 8%, Health 3%."
        ),
        "metadata": {
            "type": "guideline",
            "archetype": "single_professional",
            "col": "high",
            "household": "1",
        },
    },
    {
        "title": "Family of 3 - Medium COL",
        "content": (
            "Archetype: family of three in medium cost-of-living area.\n"
            "Baseline monthly allocation:\n"
            "- Rent: 25-35%\n- Utilities: 6-10%\n- Groceries: 10-15%\n- Transport: 8-12%\n"
            "- Health: 4-6%\n- Subscriptions: 2-4%\n- Shopping & Entertainment: 4-8%\n"
            "Elasticity: Cut order -> Shopping & Entertainment (high), Subscriptions (med), Transport/Groceries (med), Health (low).\n"
            "Floors: Groceries 10%, Health 4%."
        ),
        "metadata": {
            "type": "guideline",
            "archetype": "family_3",
            "col": "medium",
            "household": "3",
        },
    },
    {
        "title": "Student - Low COL",
        "content": (
            "Archetype: student in low cost-of-living town.\n"
            "Baseline monthly allocation:\n"
            "- Rent: 20-30%\n- Utilities: 5-8%\n- Groceries: 10-14%\n- Transport: 3-7%\n"
            "- Health: 2-4%\n- Subscriptions: 2-4%\n- Shopping & Entertainment: 6-12%\n"
            "Elasticity: Cut order -> Shopping & Entertainment (high), Subscriptions (med), Transport (med), Groceries/Health (low).\n"
            "Floors: Groceries 10%, Health 2%."
        ),
        "metadata": {
            "type": "guideline",
            "archetype": "student",
            "col": "low",
            "household": "1",
        },
    },
    {
        "title": "Emergency Fund Template",
        "content": (
            "Goal template: Emergency fund.\n"
            "Recommendation: target 3-6 months of expenses.\n"
            "Monthly allocation = target_amount / remaining_months.\n"
            "Prioritize before discretionary categories."
        ),
        "metadata": {"type": "goal_template", "goal": "emergency_fund"},
    },
    {
        "title": "Travel Fund Template",
        "content": (
            "Goal template: Travel fund.\n"
            "Set a monthly sinking fund: target_amount / remaining_months.\n"
            "Cut from Shopping & Entertainment first, then Subscriptions."
        ),
        "metadata": {"type": "goal_template", "goal": "travel"},
    },
]

# Seeding Budget FAISS

if os.path.exists(BUDGET_FAISS_PATH):
    shutil.rmtree(BUDGET_FAISS_PATH)
    os.remove("../data/budget_memory.json")
    print("🔴 Budget FAISS stores and json file deleted")
else:
    print("ℹ️  Budget FAISS stores directory doesn't exist, nothing to delete")

budget_docs = []
for g in budget_guideline_texts:
    budget_docs.append(
        Document(
            page_content=f"{g['title']}\n\n{g['content']}",
            metadata={
                "doc_id": str(uuid4()),
                **g["metadata"],
                "collection": "budget_guidelines",
            },
        )
    )


if os.path.exists(f"{BUDGET_FAISS_PATH}/index.faiss"):
    db_guidelines = FAISS.load_local(
        f"{BUDGET_FAISS_PATH}",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("🔔 Loaded existing budget guidelines FAISS")
else:
    db_guidelines = FAISS.from_documents(budget_docs, embedding_model)
    db_guidelines.save_local(f"{BUDGET_FAISS_PATH}")
    print("✅ Created and saved budget guidelines FAISS")


print(
    f"Number of guideline vectors in Budget FAISS: {len(db_guidelines.index_to_docstore_id)}"
)
size_bytes = get_folder_size(f"{BUDGET_FAISS_PATH}")
size_mb = size_bytes / (1024 * 1024)
print(f"Guidelines Budget FAISS size on disk: {size_mb:.2f} MB")


# Seeding Tranaction FAISS

if os.path.exists(TRANSACTION_FAISS_PATH):
    shutil.rmtree(TRANSACTION_FAISS_PATH)
    os.remove("../data/transaction_memory.json")
    print("🔴 Transaction FAISS stores deleted")
else:
    print("ℹ️  Transaction FAISS stores directory doesn't exist, nothing to delete")

docs = []
for example in transaction_examples:
    docs.append(
        Document(
            page_content=example["input"],
            metadata={
                "doc_id": str(uuid4()),
                "input": example["input"],
                "amount": example["amount"],
                "item_name": example["item_name"],
                "category": example["category"],
                "action": "add",
                "source": "training_data",
            },
        )
    )

split_docs = text_splitter.split_documents(docs)


if os.path.exists(f"{TRANSACTION_FAISS_PATH}/index.faiss"):
    db = FAISS.load_local(
        f"{TRANSACTION_FAISS_PATH}",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("🔔 Loaded existing transactions FAISS")
else:
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(f"{TRANSACTION_FAISS_PATH}")
    print("✅ Created and saved transactions FAISS")

print(
    f"Number of guideline vectors in Transaction FAISS: {len(db_guidelines.index_to_docstore_id)}"
)
size_bytes = get_folder_size(f"{TRANSACTION_FAISS_PATH}")
size_mb = size_bytes / (1024 * 1024)
print(f"Guidelines Transaction FAISS size on disk: {size_mb:.2f} MB")
