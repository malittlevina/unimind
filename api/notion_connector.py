

import requests
import os

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
NOTION_TOKEN = os.getenv("NOTION_API_TOKEN")

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json"
}

def create_page(database_id, title):
    url = f"{NOTION_API_BASE}/pages"
    data = {
        "parent": { "database_id": database_id },
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def query_database(database_id, filter_payload=None):
    url = f"{NOTION_API_BASE}/databases/{database_id}/query"
    response = requests.post(url, headers=headers, json=filter_payload or {})
    return response.json()

def append_block_children(block_id, children):
    url = f"{NOTION_API_BASE}/blocks/{block_id}/children"
    response = requests.patch(url, headers=headers, json={"children": children})
    return response.json()