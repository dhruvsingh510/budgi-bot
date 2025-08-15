#!/usr/bin/env python3
"""
Main entry point for the Intelligent Personal Finance Bot.

This script initializes and runs the orchestrator service which automatically
routes user requests to either the Budget or Transaction service using LangGraph
and an LLM for intelligent decision making.
"""

import os
import sys
from pathlib import Path

# Add the services directory to Python path
services_dir = Path(__file__).parent / "services"
sys.path.insert(0, str(services_dir))


def main():
    """Main entry point for the application."""
    print("🚀 Starting Intelligent Personal Finance Bot...")
    print("=" * 60)

    try:
        # Import and run the orchestrator service
        from orchestrator_service import main as orchestrator_main

        orchestrator_main()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(
            "Please ensure all dependencies are installed and services are in the correct directory."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
