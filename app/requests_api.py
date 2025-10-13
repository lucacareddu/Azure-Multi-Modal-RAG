import requests

BASE_URL = 'http://127.0.0.1:8000'
# AGENT_ID = 'your-agent-id'
HEADERS = {'Content-Type': 'application/json'}

# def create_session():
#     # url = f"{BASE_URL}/api/v1/agents/{AGENT_ID}/sessions"
#     response = requests.post(url, headers=HEADERS)
#     if response.status_code == 201:
#         return response.json().get("id")
#     return None

def send_message(message):
    url = f"{BASE_URL}/chat"
    payload = {'question': message}
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

# def accept_action(session_id):
#     url = f"{BASE_URL}/api/v1/sessions/{session_id}/actions/accept"
#     response = requests.post(url, headers=HEADERS)
#     return response.json()

# def reject_action(session_id, reason):
#     url = f"{BASE_URL}/api/v1/sessions/{session_id}/actions/reject"
#     payload = {'reason': reason}
#     response = requests.post(url, headers=HEADERS, json=payload)
#     return response.json()