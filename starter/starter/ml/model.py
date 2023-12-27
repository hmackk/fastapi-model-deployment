import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def evaluate_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature.

    Inputs
    -------
    df (pd.DataFrame): Test dataframe pre-processed with features,
                       including the categorical feature for slicing.
    feature (str): Feature on which to perform the slices.
    y (np.array): Corresponding known labels, binarized.
    preds (np.array): Predicted labels, binarized.

    Returns
    -------
    pd.DataFrame: Dataframe with columns:
        - feature value: value of the categorical feature
        - n_samples: number of data samples in the slice
        - precision: precision score
        - recall: recall score
        - fbeta: fbeta score
    """
    # Unique options for the specified feature
    slice_options = df[feature].unique()

    # Path to save the results
    save_path = os.path.join(".", "slice_output.txt")

    # Initialize the DataFrame with column names
    performance_df = pd.DataFrame(
        columns=["feature value", "n_samples", "precision", "recall", "fbeta"]
    )

    for option in slice_options:
        slice_mask = df[feature] == option
        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]

        # Calculate evaluation metrics
        precision = precision_score(slice_y, slice_preds)
        recall = recall_score(slice_y, slice_preds)
        fbeta = fbeta_score(slice_y, slice_preds, beta=1)

        # Append results to the DataFrame
        performance_df = pd.concat(
            [
                performance_df,
                pd.DataFrame(
                    [
                        {
                            "feature value": option,
                            "n_samples": len(slice_y),
                            "precision": precision,
                            "recall": recall,
                            "fbeta": fbeta,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # Save the results to a CSV file, overwriting the file each time
    performance_df.to_csv(save_path, mode="w", index=False)
    return performance_df
