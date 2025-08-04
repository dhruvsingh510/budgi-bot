# BudgiBot Frontend
Meet BudgiBot, your friendly neighbourhood budgeting bot
A modern Streamlit chatbot interface for BudgiBot with a sleek dark theme and clean UI.

## Features

- 🎨 **Modern Dark Theme** - Sleek gradient design with purple/indigo color palette
- 💬 **Chatbot Interface** - Clean message bubbles with typing indicators
- 📱 **Responsive Design** - Works great on desktop and mobile
- 🔌 **API Ready** - Easy integration with your backend
- 🚀 **Quick Actions** - Sidebar with common tasks
- 📊 **Demo Mode** - Sample responses for testing

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy environment file:

```bash
cp .env.example .env
```

3. Update `.env` with your backend URL:

```bash
API_BASE_URL=http://localhost:8000
```

## Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## API Integration

The app is ready to connect to your backend. Key endpoints expected:

- `POST /api/chat` - Send messages to the bot
- `GET /api/budget` - Get budget information
- `POST /api/budget` - Create/update budgets
- `GET /api/transactions` - Get transaction history
- `POST /api/transactions` - Add new transactions
- `GET /api/analytics` - Get spending analytics
- `GET /api/health` - Health check

## Customization

- **Colors**: Update `THEME_COLORS` in `config.py`
- **Endpoints**: Modify `ENDPOINTS` in `config.py`
- **Styling**: Edit the CSS in `app.py`

## Demo Mode

When `ENABLE_DEMO_MODE=true`, the app uses sample responses instead of making API calls. Perfect for testing the UI before connecting to your backend.
