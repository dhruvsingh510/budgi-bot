# BudgiBot

🤖 **Intelligent AI-powered personal finance assistant with automatic routing**

A modern Streamlit chatbot interface that connects to the backend's intelligent orchestrator, which automatically routes your requests to either Budget or Transaction services using AI analysis.

## Features

- 🧠 **AI-Powered Routing** - Automatically determines if your request is budget or transaction related
- 🎨 **Modern Dark Theme** - Sleek gradient design with purple/indigo color palette
- 💬 **Natural Language Interface** - Just type naturally, no commands needed
- 📱 **Responsive Design** - Works great on desktop and mobile
- 🔌 **Real-time API Integration** - Connected to intelligent backend orchestrator
- 🚀 **Quick Actions** - Sidebar with common tasks
- 📊 **AI Routing Insights** - See which service handled your request and confidence scores
- 🛡️ **Fallback Mode** - Graceful degradation when backend is unavailable

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create environment files with API key value for frontend and backend

## Running the App

Make sure your backend is running first:
```bash
# In the backend directory
python main.py
```

Then start the frontend:
```bash
# In the frontend directory
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Backend Integration

The frontend connects to the intelligent backend orchestrator at `http://localhost:8080` with these endpoints:

- `POST /chat` - Send messages to the AI-powered bot
- `GET /health` - Backend health check
- `GET /` - API information
- `GET /docs` - Interactive API documentation

## Usage Examples

The AI will automatically route your requests - just type naturally:

**Budget Planning:**

- "Set my income to ₹5000 with medium cost of living"
- "Add a goal to save ₹10000 in 12 months"
- "Show my budget plan"
- "Adjust my food budget to ₹400"

**Transaction Management:**

- "Add coffee ₹5"
- "Show recent transactions"
- "Add groceries ₹85 to food category"
- "Show all my food transactions"

## AI Routing Display

The sidebar shows:

- **API Status** - Connection to backend
- **Last AI Routing** - Which service (Budget/Transaction) handled your request
- **Confidence Score** - How confident the AI was in its decision
- **Reasoning** - Why the AI chose that service


## Troubleshooting

1. **Backend Connection Issues**: Make sure the backend is running on port 8080
2. **API Errors**: Check that your environment variables are set correctly
3. **Fallback Mode**: The app will show demo responses if the backend is unavailable
