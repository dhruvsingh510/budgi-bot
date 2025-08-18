#!/usr/bin/env python3
"""
Quick test script to verify LLM API is working
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def test_llm_connection():
    """Test if LLM API is working properly."""

    # Load environment variables
    load_dotenv()

    # Check if required environment variables exist
    groq_api_key = "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"
    model_name = "llama-3.3-70b-versatile"

    print("🔍 Checking environment variables...")
    print(f"GROQ_API_KEY: {'✅ Found' if groq_api_key else '❌ Missing'}")
    print(f"LITELLM_MODEL: {model_name or '❌ Missing'}")

    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False

    try:
        # Initialize LLM
        print("\n🚀 Initializing LLM...")
        llm = ChatGroq(
            model_name=model_name or "llama3-8b-8192",
            groq_api_key=groq_api_key,
        )
        print("✅ LLM initialized successfully")

        # Test simple API call
        print("\n💬 Testing API call...")
        test_message = "Hello! Can you respond with 'API is working'?"
        response = llm.invoke([HumanMessage(content=test_message)])
        print(response)

        print(f"📤 Sent: {test_message}")
        print(f"📥 Received: {response.content}")
        print("✅ LLM API call successful!")

        # Test JSON parsing capability
        print("\n🧮 Testing JSON parsing...")
        json_prompt = """
        Respond with valid JSON only:
        {
          "status": "working",
          "message": "LLM can parse JSON",
          "confidence": 0.95
        }
        """
        json_response = llm.invoke([HumanMessage(content=json_prompt)])
        print(f"📥 JSON Response: {json_response.content}")

        # Try to parse as JSON
        import json

        try:
            parsed = json.loads(json_response.content)
            print("✅ JSON parsing successful!")
            print(f"   Status: {parsed.get('status')}")
            print(f"   Message: {parsed.get('message')}")
            print(f"   Confidence: {parsed.get('confidence')}")
        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed, but LLM responded")

        return True

    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        print(f"   Full error: {traceback.format_exc()}")
        return False


def test_resource_manager_llm():
    """Test LLM through resource manager (as used in the app)."""
    try:
        print("\n🔗 Testing through ResourceManager...")
        from backend.services.resource_manager import resource_manager

        llm = resource_manager.llm
        response = llm.invoke([HumanMessage(content="Say 'ResourceManager LLM works'")])
        print(f"📥 ResourceManager Response: {response.content}")
        print("✅ ResourceManager LLM working!")
        return True

    except Exception as e:
        print(f"❌ ResourceManager LLM failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 LLM API Connection Test")
    print("=" * 50)

    # Test direct connection
    direct_test = test_llm_connection()

    # Test through resource manager
    resource_test = test_resource_manager_llm()

    print("\n📊 Test Results:")
    print(f"Direct LLM Test: {'✅ PASS' if direct_test else '❌ FAIL'}")
    print(f"ResourceManager Test: {'✅ PASS' if resource_test else '❌ FAIL'}")

    if direct_test and resource_test:
        print("\n🎉 All tests passed! Your LLM API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check your API key and network connection.")
