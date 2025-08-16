"""
FastAPI server for the Intelligent Personal Finance Bot.

This server exposes the orchestrator service through HTTP endpoints,
allowing clients to interact with the bot via REST API calls.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import traceback

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add services directory to path
services_dir = Path(__file__).parent.parent / "services"
sys.path.insert(0, str(services_dir))


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ..., description="User's message to the bot", min_length=1, max_length=1000
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Bot's response to the user")
    service_used: str = Field(
        ..., description="Which service handled the request (budget/transaction)"
    )
    confidence: float = Field(
        ..., description="Confidence score of the routing decision"
    )
    reasoning: str = Field(
        ..., description="Explanation for why this service was chosen"
    )
    status: str = Field(default="success", description="Status of the request")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    status: str = Field(default="error", description="Status of the request")


# Initialize FastAPI app
app = FastAPI(
    title="Personal Finance Bot API",
    description="Intelligent personal finance bot with automatic routing to budget and transaction services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instance
bot_instance = None


def initialize_bot():
    """Initialize the bot instance."""
    global bot_instance
    try:
        from orchestrator_service import BudgetBot

        bot_instance = BudgetBot()
        print("✅ Bot initialized successfully for API server")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize bot: {e}")
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the bot when the server starts."""
    print("🚀 Starting Personal Finance Bot API Server...")
    print("⏳ Initializing AI services (this may take a moment)...")

    if not initialize_bot():
        print("❌ Failed to initialize bot - server may not work properly")
        print("🔍 Check your environment variables (GROQ_API_KEY, LITELLM_MODEL)")
        print("🔍 Ensure all dependencies are installed")
    else:
        print("✅ All services initialized successfully!")
        print("🤖 Bot is ready to handle requests")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Personal Finance Bot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "POST /chat - Send a message to the bot",
            "health": "GET /health - Check server health",
            "docs": "GET /docs - API documentation",
        },
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    global bot_instance

    if bot_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot not initialized",
        )

    return {
        "status": "healthy",
        "bot_status": "initialized",
        "message": "API server is running and bot is ready",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(request: ChatRequest):
    """
    Send a message to the bot and get a response.

    The bot will automatically route your request to either the Budget or Transaction service
    based on the content of your message using AI analysis.

    Examples:
    - "Set my income to $5000" → Budget service
    - "Add coffee $5" → Transaction service
    - "Show my budget plan" → Budget service
    - "Show recent transactions" → Transaction service
    """
    global bot_instance

    if bot_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot not initialized - please check server logs",
        )

    try:
        print(f"📥 API Request: {request.message}")

        # Process the request through the orchestrator
        response = bot_instance.process_request(request.message)

        # Get routing information from the last state (if available)
        try:
            last_state = getattr(bot_instance, "last_routing_state", None)
            if last_state:
                service_used = last_state.get("service_route", "unknown")
                confidence = last_state.get("confidence", 0.0)
                reasoning = last_state.get(
                    "reasoning", "No routing information available"
                )
            else:
                service_used = "unknown"
                confidence = 0.0
                reasoning = "Routing information not available"
        except:
            service_used = "unknown"
            confidence = 0.0
            reasoning = "Could not retrieve routing information"

        print(f"📤 API Response: {response[:100]}...")

        return ChatResponse(
            response=response,
            service_used=service_used,
            confidence=confidence,
            reasoning=reasoning,
            status="success",
        )

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your request: {str(e)}",
        )


def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Run the FastAPI server."""
    import socket

    # Check if port is available
    def is_port_available(host: str, port: int) -> bool:
        """Check if a port is available on the given host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(
                    (host if host != "0.0.0.0" else "localhost", port)
                )
                return result != 0
        except Exception:
            return False

    # Check port availability
    if not is_port_available("localhost", port):
        print(f"❌ Port {port} is already in use!")
        print(f"💡 Try a different port: python main.py --port {port + 1}")

        # Suggest alternative ports
        for alt_port in range(port + 1, port + 10):
            if is_port_available("localhost", alt_port):
                print(f"✅ Port {alt_port} is available")
                break

        sys.exit(1)

    print(f"🌐 Starting API server on http://{host}:{port}")
    print(f"📚 API documentation available at http://{host}:{port}/docs")
    print(f"🏥 Health check available at http://{host}:{port}/health")
    print(f"📋 API info available at http://{host}:{port}/")

    try:
        uvicorn.run(
            "api.server:app", host=host, port=port, reload=reload, log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
