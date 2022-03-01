import pandas as pd
import joblib
from joblib import dump
from ml.helper_functions import *
from ml.clean_dataset import *
from sklearn.model_selection import train_test_split

def check_score():
    """
    function for slicing the data and checking the scores of each slice
    :return:
    txt file is saved with the scores of each slice
    """
    df = pd.read_csv('../../data/cleaned_dataset.csv')
    _,test = train_test_split(df,test_size=0.2)

    trained_model = joblib.load("../../model/trained_model.joblib")
    encoder = joblib.load( "../../model/encoder.joblib")
    lb = joblib.load("../../model/lb.joblib")

    slice_values = []

    for cat in ml.helper_functions.get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat]==cls]
            X_test,y_test,_,_ = ml.helper_functions.process_data(df_temp,
                                                                 categorical_features=ml.helper_functions.get_cat_features(),
                                             label="salary", training=False, encoder=encoder, lb=lb)

            y_preds = trained_model.predict(X_test)
            fb, rc, prc = get_model_metrics(y_test, y_preds)
            line = "[%s - %s] -- Precision:%s Recall:%s F-beta:%s " %(cat , cls, prc, rc, fb )
            logging.info(line)
            slice_values.append(line)

    with open('../data/model/slice_scores.txt', 'w') as out:
        logging.info('saving metrics for each slice')
        for each_slice in slice_values:
            out.write(each_slice +'\n')