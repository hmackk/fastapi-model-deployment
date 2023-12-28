# Script to train machine learning model.

import os
import pickle

import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, evaluate_slices, inference, train_model
from sklearn.model_selection import train_test_split

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")
data.columns = data.columns.str.replace(" ", "")  # remove white spaces.
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open(os.path.join("starter/model", "trained_model.pkl"), "wb"))
pickle.dump(encoder, open(os.path.join("starter/model", "encoder.pkl"), "wb"))
pickle.dump(lb, open(os.path.join("starter/model", "labelizer.pkl"), "wb"))


preds = inference(model, X_test)


precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")

for feature in cat_features:
    evaluate_slices(test, feature, y_test, preds)
