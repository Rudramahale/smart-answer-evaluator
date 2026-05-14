import requests

data = {
    "question_id": 1,
    "student_answer": "It."
}

# Send the request to your FastAPI server
response = requests.post("http://127.0.0.1:8000/check", json=data)

# Print the output!
print("Evaluated Marks:", response.json())
