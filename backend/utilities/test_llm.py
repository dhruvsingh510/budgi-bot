import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def test_llm_connection():
    """Test if LLM API is working properly."""

    load_dotenv()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    model_name = os.environ.get("LITELLM_MODEL")

    print("🔍 Checking environment variables...")
    print(f"GROQ_API_KEY: {'✅ Found' if groq_api_key else '❌ Missing'}")
    print(f"LITELLM_MODEL: {model_name or '❌ Missing'}")

    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False

    try:
        print("\n🚀 Initializing LLM...")
        llm = ChatGroq(
            model_name=model_name,
            groq_api_key=groq_api_key,
        )
        print("✅ LLM initialized successfully")

        print("\n💬 Testing API call...")
        test_message = "Hello! Can you respond with 'API is working'?"
        response = llm.invoke([HumanMessage(content=test_message)])
        print(response)

        print(f"📤 Sent: {test_message}")
        print(f"📥 Received: {response.content}")
        print("✅ LLM API call successful!")

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


if __name__ == "__main__":
    print("🧪 LLM API Connection Test")
    print("=" * 50)

    direct_test = test_llm_connection()

    print("\n📊 Test Results:")
    print(f"Direct LLM Test: {'✅ PASS' if direct_test else '❌ FAIL'}")
