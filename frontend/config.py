import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the BudgiBot frontend"""

    # Streamlit Page Config
    PAGE_TITLE = "BudgiBot - Your Budgeting Assistant"
    PAGE_ICON = "💰"

    # API Configuration - Force correct port
    API_BASE_URL = "http://localhost:8080"  # Force port 8080 for backend
    API_TIMEOUT = 30

    # Endpoints (matching the actual backend endpoints)
    ENDPOINTS = {
        "chat": "/chat",
        "health": "/health",
        "root": "/",
        "docs": "/docs",
    }

    # UI Configuration
    THEME_COLORS = {
        "primary": "#6366f1",
        "secondary": "#8b5cf6",
        "background": "#0f0f23",
        "surface": "#1a1a2e",
        "text": "#e2e8f0",
        "text_muted": "#94a3b8",
        "border": "#334155",
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
    }

    # Chat Configuration
    MAX_MESSAGE_LENGTH = 1000
    TYPING_DELAY = 1.0  # seconds

    # Demo Configuration
    ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "true").lower() == "true"
