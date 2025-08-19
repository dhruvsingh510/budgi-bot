import streamlit as st
import requests
import json
from datetime import datetime
import time
from config import Config
from api_client import APIClient

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load external CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Sample responses for demo
SAMPLE_RESPONSES = {
    "hello": "Hello! I'm BudgiBot, your friendly budgeting assistant. I can help you track expenses, set budgets, and analyze your spending patterns. How can I help you today?",
    "budget": "I can help you create and manage budgets! Here are some options:\n• Set monthly budget limits\n• Track spending by category\n• Get budget alerts\n• View budget vs actual spending\n\nWhat would you like to do?",
    "expenses": "Let me help you track your expenses! You can:\n• Add new transactions\n• View spending history\n• Categorize expenses\n• Get spending insights\n\nWould you like to add a new expense or view your spending history?",
    "analytics": "Here's your spending analytics:\n\n📊 **This Month's Overview:**\n• Total Spent: ₹1,247.50\n• Budget Remaining: ₹752.50\n• Top Category: Food & Dining (₹320)\n\n📈 **Trends:**\n• 15% increase from last month\n• Most expensive day: Friday\n• Average daily spending: ₹41.58\n\nWould you like to see more detailed analytics?",
    "default": "I'm here to help with your budgeting needs! You can ask me about:\n• Setting up budgets\n• Tracking expenses\n• Spending analytics\n• Financial goals\n• Budget alerts\n\nWhat would you like to know?",
}


# Initialize API client
@st.cache_resource
def get_api_client():
    """Get cached API client instance"""
    return APIClient()


def get_bot_response(user_message):
    """Get response from bot using real API"""
    try:
        api_client = get_api_client()

        # Try to get response from real API
        response = api_client.send_message(user_message)

        if response and response.get("status") == "success":
            bot_response = response.get("response", "I couldn't generate a response.")
            service_used = response.get("service_used", "unknown")
            confidence = response.get("confidence", 0.0)
            reasoning = response.get("reasoning", "")

            # Add metadata to session state for display
            if "last_api_info" not in st.session_state:
                st.session_state.last_api_info = {}

            st.session_state.last_api_info = {
                "service_used": service_used,
                "confidence": confidence,
                "reasoning": reasoning,
            }

            return bot_response
        else:
            # Fallback to demo mode if API fails
            return get_demo_response(user_message)

    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return get_demo_response(user_message)


def get_demo_response(user_message):
    """Fallback demo responses when API is unavailable"""
    message_lower = user_message.lower()

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

    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 Budgi Bot</h1>
        <p>Your friendly neighbourhood transaction and budgetting assistant</p>
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
            key=f"user_input_{st.session_state.input_key}",
            placeholder="Ask naturally: 'Set my budget profile' or 'Add coffee ₹5'...",
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

            # Clear input by incrementing the key (creates new widget)
            st.session_state.input_key += 1
            st.rerun()

    # Sidebar with quick actions
    with st.sidebar:
        st.markdown("### Quick Actions")

        if st.button("📊 Show Transactions", use_container_width=True):
            user_message = "Show recent transactions"
            st.session_state.messages.append({"role": "user", "content": user_message})
            bot_response = get_bot_response(user_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        if st.button("💰 Show Budget Plan", use_container_width=True):
            user_message = "Show my budget plan"
            st.session_state.messages.append({"role": "user", "content": user_message})
            bot_response = get_bot_response(user_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        if st.button("💳 Add Transaction", use_container_width=True):
            user_message = "How do I add a transaction?"
            st.session_state.messages.append({"role": "user", "content": user_message})
            bot_response = get_bot_response(user_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

        st.markdown("---")
        st.markdown("### API Status")

        # Check API connectivity
        api_client = get_api_client()
        is_healthy = api_client.check_health()
        api_status = "🟢 Connected" if is_healthy else "🔴 Disconnected"
        st.markdown(f"**Backend:** {api_status}")

        if is_healthy:
            # Show API info
            api_info = api_client.get_api_info()
            if api_info:
                st.markdown(f"**Version:** {api_info.get('version', 'Unknown')}")

        # Show last AI routing info
        if "last_api_info" in st.session_state and st.session_state.last_api_info:
            info = st.session_state.last_api_info
            st.markdown("---")
            st.markdown("### Last AI Routing")
            st.markdown(f"**Service:** {info.get('service_used', 'unknown').title()}")
            st.markdown(f"**Confidence:** {info.get('confidence', 0):.2f}")
            with st.expander("Reasoning"):
                st.markdown(info.get("reasoning", "No reasoning available"))

        st.markdown("---")
        st.markdown("### Available Endpoints")
        for name, endpoint in Config.ENDPOINTS.items():
            status_emoji = "🟢" if is_healthy else "🔴"
            st.markdown(f"• {status_emoji} `{name}`: `{endpoint}`")


if __name__ == "__main__":
    main()
