import requests
import json

# The URL of your running local API
url = "http://127.0.0.1:8000/predict"

# Sample patient data (matches the columns your model expects)
payload = {
    "Age": 30,
    "Gender": "Male",
    "Country": "United States",
    "self_employed": "No",
    "family_history": "Yes",
    "work_interfere": "Often",
    "no_employees": 63,
    "remote_work": "No",
    "tech_company": "Yes",
    "benefits": "Yes",
    "care_options": "Yes",
    "wellness_program": "No",
    "seek_help": "No",
    "anonymity": "Yes",
    "leave": "Very easy",
    "mental_health_consequence": "No",
    "phys_health_consequence": "No",
    "coworkers": "Some of them",
    "supervisor": "Yes",
    "mental_health_interview": "No",
    "phys_health_interview": "Maybe",
    "mental_vs_physical": "Yes",
    "obs_consequence": "No"
}

try:
    print("Sending request to API...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("\n✅ Success!")
        print("Response from Server:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ Error {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"\n❌ Connection Error: {e}")