import requests

X = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States",
}

y = "[' <=50K']"

r = requests.post(
    "https://fastapi-model-deployment.onrender.com/predict",
    json=X,
)

print(f"status code: {r.status_code}")
print(f"prediction: {r.json()['predictions']}, \t ground truth: {y}")
