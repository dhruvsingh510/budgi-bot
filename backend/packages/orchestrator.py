from budget_agent import BudgetAgent

# Create agent instance
agent = BudgetAgent()

# Call the agent
text, final = agent.call("show me my plan")
print(text)

# Access context
ctx = agent.get_context()
print(ctx)

# Update context
agent.set_context({"profile": None, "goals": [], "plan": None})