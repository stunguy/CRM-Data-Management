import requests

def get_chatbot_response(message):
    print(f"Send:{message}")
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {"sender":"user","message":message}
    response = requests.post(rasa_url,json=payload)
    print(f"Recieve:{response.json()}")
    return response.json()