import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="BudgiBot - Your Budgeting Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load external CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Configuration
class Config:
    API_BASE_URL = "http://localhost:8000"  # Change this to your backend URL
    ENDPOINTS = {
        "chat": "/api/chat",
        "budget": "/api/budget",
        "transactions": "/api/transactions",
        "analytics": "/api/analytics",
    }


# Sample responses for demo
SAMPLE_RESPONSES = {
    "hello": "Hello! I'm BudgiBot, your friendly budgeting assistant. I can help you track expenses, set budgets, and analyze your spending patterns. How can I help you today?",
    "budget": "I can help you create and manage budgets! Here are some options:\n• Set monthly budget limits\n• Track spending by category\n• Get budget alerts\n• View budget vs actual spending\n\nWhat would you like to do?",
    "expenses": "Let me help you track your expenses! You can:\n• Add new transactions\n• View spending history\n• Categorize expenses\n• Get spending insights\n\nWould you like to add a new expense or view your spending history?",
    "analytics": "Here's your spending analytics:\n\n📊 **This Month's Overview:**\n• Total Spent: $1,247.50\n• Budget Remaining: $752.50\n• Top Category: Food & Dining ($320)\n\n📈 **Trends:**\n• 15% increase from last month\n• Most expensive day: Friday\n• Average daily spending: $41.58\n\nWould you like to see more detailed analytics?",
    "default": "I'm here to help with your budgeting needs! You can ask me about:\n• Setting up budgets\n• Tracking expenses\n• Spending analytics\n• Financial goals\n• Budget alerts\n\nWhat would you like to know?",
}


def call_api(endpoint, data=None):
    """Make API call to backend"""
    try:
        url = f"{Config.API_BASE_URL}{Config.ENDPOINTS.get(endpoint, endpoint)}"
        if data:
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


def get_bot_response(user_message):
    """Get response from bot (simulated for now)"""
    message_lower = user_message.lower()

    # Simulate API call
    # response = call_api("chat", {"message": user_message})
    # if response:
    #     return response.get("response", "I'm sorry, I couldn't process that request.")

    # For demo purposes, use sample responses
    if any(word in message_lower for word in ["hello", "hi", "hey"]):
        return SAMPLE_RESPONSES["hello"]
    elif any(word in message_lower for word in ["budget", "budgeting"]):
        return SAMPLE_RESPONSES["budget"]
    elif any(
        word in message_lower for word in ["expense", "spending", "spend", "cost"]
    ):
        return SAMPLE_RESPONSES["expenses"]
    elif any(
        word in message_lower
        for word in ["analytics", "analysis", "report", "insights"]
    ):
        return SAMPLE_RESPONSES["analytics"]
    else:
        return SAMPLE_RESPONSES["default"]


def display_typing_indicator():
    """Display typing indicator"""
    st.markdown(
        """
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>💰 BudgiBot</h1>
        <p>Your friendly neighbourhood budgeting assistant</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            placeholder="Ask me about budgeting, expenses, or analytics...",
            label_visibility="collapsed",
        )

    with col2:
        send_button = st.button("Send", use_container_width=True)

    # Handle user input
    if (user_input and user_input.strip()) or send_button:
        if user_input.strip():
            # Add user message
            st.session_state.messages.append(
                {"role": "user", "content": user_input.strip()}
            )

            # Show typing indicator
            with st.spinner(""):
                display_typing_indicator()
                time.sleep(1)  # Simulate processing time

            # Get bot response
            bot_response = get_bot_response(user_input.strip())
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

            # Clear input
            st.session_state.user_input = ""
            st.rerun()

    # Sidebar with quick actions
    with st.sidebar:
        st.markdown("### Quick Actions")

        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "Show me my analytics"}
            )
            bot_response = get_bot_response("analytics")
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        if st.button("💰 Set Budget", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "Help me set a budget"}
            )
            bot_response = get_bot_response("budget")
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        if st.button("💳 Add Expense", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "I want to add an expense"}
            )
            bot_response = get_bot_response("expenses")
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        st.markdown("---")
        st.markdown("### API Status")

        # Check API connectivity
        api_status = "🟢 Connected" if call_api("health") else "🔴 Disconnected"
        st.markdown(f"**Backend:** {api_status}")

        st.markdown("---")
        st.markdown("### Available Endpoints")
        for name, endpoint in Config.ENDPOINTS.items():
            st.markdown(f"• `{name}`: `{endpoint}`")


if __name__ == "__main__":
    main()
