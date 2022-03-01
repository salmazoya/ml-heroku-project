import logging
import numpy as np
import pandas as pd


def clean_data(df):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [x.lstrip() for x in list(df.columns)]
    df = df[['age','workclass','education','marital-status','occupation','relationship','race','sex','native-country','hours-per-week','salary']]
    df = df.replace({' ?': None})
    df = df.replace(to_replace='None', value=np.nan).dropna()
    return df


def execute_cleaning():
    logging.info('reading the file..')
    df = pd.read_csv('../../data/census.csv')
    df = clean_data(df)
    df.to_csv('../../data/cleaned_dataset.csv', index=False)
    logging.info('saving the file..')