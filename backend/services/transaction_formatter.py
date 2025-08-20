import re
from typing import Dict, List, Any
from datetime import datetime


class TransactionFormatter:
    """Handles clean formatting of transaction data for frontend display."""

    @classmethod
    def format_amount(cls, amount: Any) -> str:
        """Format amount for display."""
        if amount is None or amount == "":
            return "Unknown amount"

        try:
            amount_float = float(amount)
            return f"₹{amount_float:.2f}"
        except (ValueError, TypeError):
            return "Unknown amount"

    @classmethod
    def format_date(cls, date_str: str) -> str:
        """Format date for display."""
        if not date_str or date_str == "Unknown date":
            return "Unknown date"

        try:
            # Extract just the date part (YYYY-MM-DD)
            date_part = date_str.split(" ")[0]
            return date_part
        except:
            return date_str

    @classmethod
    def format_single_transaction(cls, transaction: Dict) -> str:
        """Format a single transaction for clean display."""
        item_name = transaction.get("item_name", "Unknown item")
        amount = cls.format_amount(transaction.get("amount"))
        category = transaction.get("category", "Miscellaneous")
        date = cls.format_date(transaction.get("datetime", ""))

        if item_name and "•" in item_name:
            # Handle corrupted item names like "•catfood−12.00•*catfood*−12.00"
            item_name = re.split(r"[•−*]", item_name)[0].strip()

        return f"    • {item_name} - {amount} ({category}) - {date}"

    @classmethod
    def format_transaction_list(cls, transactions: List[Dict]) -> str:
        """Format a list of transactions for display."""
        if not transactions:
            return "📊 No transactions found."

        # Sort by date (most recent first)
        sorted_transactions = sorted(
            transactions, key=lambda x: x.get("datetime", ""), reverse=True
        )

        formatted_lines = []
        for transaction in sorted_transactions:
            formatted_lines.append(cls.format_single_transaction(transaction))

        return "\n".join(formatted_lines)

    @classmethod
    def format_category_summary(cls, category_name: str, category_data: Dict) -> str:
        """Format category summary with clean styling."""
        count = category_data.get("count", 0)
        total = category_data.get("total", 0)

        return f"💰 **{category_name}**: {count} transactions, Total: ₹{total:.2f}\n"

    @classmethod
    def format_all_transactions_response(cls, grouped_data: Dict) -> str:
        """Format the complete transactions by category response."""
        if not grouped_data:
            return "📊 No transactions found."

        lines = ["📊 **All Transactions by Category**", ""]
        total_transactions = 0
        total_amount = 0.0

        # Sort categories by transaction count (descending)
        sorted_categories = sorted(
            grouped_data.items(), key=lambda x: x[1].get("count", 0), reverse=True
        )

        for category_name, category_data in sorted_categories:
            count = category_data.get("count", 0)
            category_total = category_data.get("total", 0)
            transactions = category_data.get("transactions", [])

            total_transactions += count
            total_amount += category_total

            # Add category header
            lines.append(cls.format_category_summary(category_name, category_data))

            # Add all transactions for this category
            if transactions:
                transaction_list = cls.format_transaction_list(transactions)
                lines.append(transaction_list)
            else:
                lines.append("📊 No transactions found.")

            # Add spacing between categories
            lines.append("")

        # Add summary
        lines.append("📈 **Summary**")
        lines.append(
            f"Total: {total_transactions} transactions across {len(grouped_data)} categories"
        )
        lines.append(f"Grand Total: ₹{total_amount:.2f}")

        return "\n".join(lines)

    @classmethod
    def format_search_results(
        cls, transactions: List[Dict], search_type: str, category: str = ""
    ) -> str:
        """Format search results with proper headers."""
        if not transactions:
            if search_type == "category":
                return f"📋 No transactions found in {category} category."
            elif search_type == "recent":
                return "📋 No similar transactions found."
            else:
                return "📋 No transactions found."

        if search_type == "category":
            header = (f"📋 **Found {len(transactions)} transactions in {category} category:**")
        elif search_type == "recent":
            header = f"📋 **Found {len(transactions)} similar recent transactions:**"
        else:
            header = f"📋 **Found {len(transactions)} transactions:**"

        transaction_list = cls.format_transaction_list(transactions)
        return header + "\n\n" + transaction_list
