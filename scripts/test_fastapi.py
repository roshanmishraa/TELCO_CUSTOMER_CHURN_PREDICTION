import requests

URL = "http://127.0.0.1:8000/predict"

# Sample payload matching EXACT CustomerData schema
sample_data = {
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 5,
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

response = requests.post(URL, json=sample_data)

print("\nStatus Code:", response.status_code)
print("API Response:", response.json())
