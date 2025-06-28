

import requests

class ExternalAIBridge:
    def __init__(self, service_url):
        self.service_url = service_url

    def send_prompt(self, prompt, model="default", temperature=0.7):
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature
        }
        try:
            response = requests.post(f"{self.service_url}/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"External AI error: {e}")
            return None

    def get_supported_models(self):
        try:
            response = requests.get(f"{self.service_url}/models")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve model list: {e}")
            return []