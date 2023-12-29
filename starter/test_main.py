import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def class_0():
    """
    Class 0 fixture.
    """
    return {
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


@pytest.fixture
def class_1():
    """
    Class 1 fixture.
    """
    return {
        "age": 52,
        "workclass": " Self-emp-not-inc",
        "fnlgt": 209642,
        "education": " HS-grad",
        "education-num": 9,
        "marital-status": " Married-civ-spouse",
        "occupation": " Exec-managerial",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": " United-States",
        "salary": " >50K",
    }


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello! This is a welcome message!"}


def test_post_class_0(class_0):
    response = client.post("/predict", json=class_0)
    assert response.status_code == 200
    assert response.json() == {"predictions": "[' <=50K']"}


def test_post_class_1(class_1):
    response = client.post("/predict", json=class_1)
    assert response.status_code == 200
    assert response.json() == {"predictions": "[' >50K']"}
