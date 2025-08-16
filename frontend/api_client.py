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
                # Only show errors in console, not in UI
                print(f"API Error: {url} - {method.upper()} - {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            # Only show errors in console, not in UI
            print(f"Request failed: {e}")
            return None

    def send_message(self, message: str) -> Optional[Dict]:
        """Send a chat message to the bot"""
        data = {"message": message}
        return self._make_request("POST", Config.ENDPOINTS["chat"], data=data)

    def check_health(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}{Config.ENDPOINTS['health']}", timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def get_api_info(self) -> Optional[Dict]:
        """Get API information from root endpoint"""
        return self._make_request("GET", Config.ENDPOINTS["root"])

    def get_detailed_health(self) -> Optional[Dict]:
        """Get detailed health information"""
        return self._make_request("GET", Config.ENDPOINTS["health"])
