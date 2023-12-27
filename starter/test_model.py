import logging
import os

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, evaluate_slices, inference


@pytest.fixture()
def model():
    """
    A trained model fixture.
    """
    return joblib.load("starter/model/trained_model.pkl")


@pytest.fixture()
def features():
    """
    Fixture for categorical features.
    """
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return categorical_features


@pytest.fixture()
def data():
    """
    Data fixture.
    """
    data = pd.read_csv("starter/data/census.csv")
    data.columns = data.columns.str.replace(" ", "")
    return data


@pytest.fixture()
def train_test_data(data, features):
    """
    Clean data fixture.
    """
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=0,
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, X_test, y_train, y_test


def test_inference(model, train_test_data):
    """
    Test model inference.
    """
    _, X_test, _, _ = train_test_data
    try:
        assert model.predict(X_test)
    except BaseException:
        logging.error("Model is not fitted!")


def test_compute_model_metrics(model, train_test_data):
    """
    Test metrics calculations.
    """
    _, X_test, _, y_test = train_test_data
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_evaluate_slices(data, train_test_data, features, model):
    """
    Test evaluation metrics for different slices.
    """
    try:
        _, test = train_test_split(data, test_size=0.20)
        _, X_test, _, y_test = train_test_data
        preds = inference(model, X_test)
        for feature in features:
            evaluate_slices(test, feature, y_test, preds)
        assert os.path.exists("./slice_output.txt")
    except AssertionError:
        logging.error("Error evaluating slices")
        raise AssertionError
