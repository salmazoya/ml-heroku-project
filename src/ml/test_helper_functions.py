import pandas as pd
import pytest
import joblib
from joblib import dump
from .helper_functions import *

@pytest.fixture
def data():
    df = pd.read_csv('../../data/cleaned_dataset.csv')
    return df

def test_X_y_splits(data):

    encoder = joblib.load("../../model/encoder.joblib")
    lb = joblib.load("../../model/lb.joblib")

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features= get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)

def test_process_encoder(data):

    encoder1 = joblib.load("../../model/encoder.joblib")
    lb1 = joblib.load("../../model/lb.joblib")

    _, _, encoder, lb = process_data(
        data,
        categorical_features= get_cat_features(),
        label="salary", training=True)

    _, _, _, _ = process_data(
        data,
        categorical_features = get_cat_features(),
        label="salary", encoder = encoder1, lb=lb1, training=False)

    assert encoder.get_params() == encoder1.get_params()
    assert lb.get_params() == lb1.get_params()

def test_inference_above():
    """
    Check inference performance
    """
    model = joblib.load("../../model/trained_model.joblib")
    encoder = joblib.load("../../model/encoder.joblib")
    lb = joblib.load("../../model/lb.joblib")

    array = np.array([[
                     34,
                     " Private",
                      " Doctorate",
                     " Married-civ-spouse",
                     " Tech-support",
                     " Wife",
                     "Black",
                     " Male",
                     80,
                     " United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
                df_temp,
                categorical_features = get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == " >50K"


def test_inference_below():
    """
    Check inference performance
    """
    model = joblib.load("../../model/trained_model.joblib")
    encoder = joblib.load("../../model/encoder.joblib")
    lb = joblib.load("../../model/lb.joblib")

    array = np.array([[
                     20,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Female",
                     40,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
                df_temp,
                categorical_features= get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == " <=50K"