

import requests
import json

class DaemonGateway:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def send_command(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def query_status(self) -> dict:
        return self.send_command("status", {})

    def invoke_scroll(self, scroll_name: str, context: dict) -> dict:
        return self.send_command("invoke_scroll", {"scroll": scroll_name, "context": context})

    def unimind_query(self, prompt: str) -> dict:
        return self.send_command("unimind_query", {"prompt": prompt})

# Example usage
if __name__ == "__main__":
    gateway = DaemonGateway()
    result = gateway.query_status()
    print("Daemon Status:", result)