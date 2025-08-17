#!/usr/bin/env python3
"""
Main entry point for the Intelligent Personal Finance Bot.

This script can run either:
1. API Server mode (default): Exposes HTTP API on port 8080
2. Console mode: Interactive command-line interface
"""

import argparse
import sys
from pathlib import Path


def run_console():
    """Run the interactive console interface."""
    print("🚀 Starting Personal Finance Bot Console...")
    print("=" * 60)

    # Add the services directory to Python path
    services_dir = Path(__file__).parent / "services"
    sys.path.insert(0, str(services_dir))

    try:
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


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Intelligent Personal Finance Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
          Examples:
            python main.py              # Run API server on port 8080
            python main.py --console    # Run interactive console
            python main.py --port 3000  # Run API server on custom port
        """,
    )

    parser.add_argument(
        "--console",
        action="store_true",
        help="Run in console mode instead of API server",
    )

    parser.add_argument(
        "--port", type=int, default=8080, help="Port for API server (default: 8080)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for API server (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    if args.console:
        run_console()
    else:
        print(f"🌐 API Server will start on http://{args.host}:{args.port}")
        print(f"📚 Documentation available at http://{args.host}:{args.port}/docs")
        print("💡 Use --console flag to run in interactive mode")
        print()

        try:
            from api.server import run_server

            run_server(host=args.host, port=args.port, reload=False)
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print("Please install FastAPI dependencies: pip install fastapi uvicorn")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
