import requests
import json
from typing import Dict, Any, Optional
from config import Config

class APIClient:
    """Client for making API calls to the BudgiBot backend"""

    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = base_url or Config.API_BASE_URL
        self.timeout = timeout or Config.API_TIMEOUT
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def _make_request(
        self, method: str, endpoint: str, data: Dict = None, params: Dict = None
    ) -> Optional[Dict]:
        """Make HTTP request to the API"""
        try:
            url = f"{self.base_url}{endpoint}"

            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def send_message(self, message: str, user_id: str = None) -> Optional[Dict]:
        """Send a chat message to the bot"""
        data = {"message": message}
        if user_id:
            data["user_id"] = user_id
        return self._make_request("POST", Config.ENDPOINTS["chat"], data=data)

    def get_budget(self, user_id: str = None) -> Optional[Dict]:
        """Get user's budget information"""
        params = {"user_id": user_id} if user_id else None
        return self._make_request("GET", Config.ENDPOINTS["budget"], params=params)

    def create_budget(self, budget_data: Dict) -> Optional[Dict]:
        """Create a new budget"""
        return self._make_request("POST", Config.ENDPOINTS["budget"], data=budget_data)

    def get_transactions(self, user_id: str = None, limit: int = 50) -> Optional[Dict]:
        """Get user's transactions"""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        return self._make_request(
            "GET", Config.ENDPOINTS["transactions"], params=params
        )

    def add_transaction(self, transaction_data: Dict) -> Optional[Dict]:
        """Add a new transaction"""
        return self._make_request(
            "POST", Config.ENDPOINTS["transactions"], data=transaction_data
        )

    def get_analytics(
        self, user_id: str = None, period: str = "month"
    ) -> Optional[Dict]:
        """Get spending analytics"""
        params = {"period": period}
        if user_id:
            params["user_id"] = user_id
        return self._make_request("GET", Config.ENDPOINTS["analytics"], params=params)

    def check_health(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}{Config.ENDPOINTS['health']}", timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile information"""
        return self._make_request("GET", f"{Config.ENDPOINTS['user']}/{user_id}")

    def get_goals(self, user_id: str = None) -> Optional[Dict]:
        """Get user's financial goals"""
        params = {"user_id": user_id} if user_id else None
        return self._make_request("GET", Config.ENDPOINTS["goals"], params=params)

    def create_goal(self, goal_data: Dict) -> Optional[Dict]:
        """Create a new financial goal"""
        return self._make_request("POST", Config.ENDPOINTS["goals"], data=goal_data)
