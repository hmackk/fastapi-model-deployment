# Model Card

## Model Details

The model used for prediction is a RandomForest Classifier from the scikit-learn library with the defualt hyperparmeters.

## Intended Use

This model demonstrates the capability to forecast an individual's salary level by leveraging different features.
Training Data

The training data is from the UCI Machine Learning Repository. The data is from the 1994 Census database. The data was collected by Barry Becker from the 1994 Census database. The data set contains 48,842 instances and 14 attributes. It is available at https://archive.ics.uci.edu/ml/datasets/census+income.
## Evaluation Data
The model was evaluated on a test set which consisted of 20% of the total data.

## Metrics

The model achieved the following scores:

Precision: 0.71
Recall: 0.62
Fbeta: 0.66

## Ethical Considerations

Exercise caution when assessing and analyzing model performance, given that the training features encompassed information related to race and sex, potentially resulting in biased predictions.

Cautions and Recommendations

Limit the application of this model solely to predicting an individual's salary level, and refrain from utilizing it for any other purposes.
