

import datetime
import requests

class CalendarSync:
    def __init__(self, calendar_api_url, auth_token):
        self.calendar_api_url = calendar_api_url
        self.auth_token = auth_token

    def fetch_events(self, start_date, end_date):
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        params = {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }
        response = requests.get(f"{self.calendar_api_url}/events", headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching events: {response.status_code}")
            return []

    def create_event(self, title, start_time, end_time, description=""):
        headers = {"Authorization": f"Bearer {self.auth_token}", "Content-Type": "application/json"}
        data = {
            "title": title,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "description": description
        }
        response = requests.post(f"{self.calendar_api_url}/events", headers=headers, json=data)
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Error creating event: {response.status_code}")
            return None