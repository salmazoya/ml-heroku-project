# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from joblib import dump
import joblib
from ml.helper_functions import *


def train_test_model():

    data = pd.read_csv('../../data/cleaned_dataset.csv')
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = ml.helper_functions.process_data(
        train, categorical_features=ml.helper_functions.get_cat_features(), label="salary", training=True)
    trained_model = ml.helper_functions.training_model(X_train,y_train)

    joblib.dump(trained_model,"../../model/trained_model.joblib")
    joblib.dump(encoder , "../../model/encoder.joblib")
    joblib.dump(lb, "../../model/lb.joblib")
