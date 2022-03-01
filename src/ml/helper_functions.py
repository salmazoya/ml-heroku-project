""" helper functions script """
import logging
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
"""
Common functions module
"""
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def get_cat_features():
    """Returns the categorical features"""
    cat_features=["workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    return cat_features

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def training_model(X_train,y_train):
    """
    Trains the machine learning model and returns the model

    Input :
    ------
    :param X_train: training data
    :param y_train: labels

    Output:
    -----
    Returns: model
    """

    model = RandomForestClassifier(random_state=8, max_depth=16, n_estimators=128)
    print('Random Forest model selected')
    model.fit(X_train,y_train)


    return model

def get_model_metrics(y,preds):
    """ get the model metrics
    Input:
    -----
    y: actual labels , np.array
    preds: predicted labels, np.array

    Output:
    ----
    fbeta: float
    precision: float
    recall: float
    """
    fbeta = fbeta_score(y,preds,beta=1,zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    precision = precision_score(y,preds,zero_division=1)
    return fbeta, recall, precision

def inference(model,X):
    """ Run model inferences and return the predictions.

        Inputs
        ------
        model : trained model
            Trained machine learning model.
        X : np.array
            Data used for prediction.
        Returns
        -------
        y_preds : np.array
            Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds

