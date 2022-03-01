"""
Testing the cleaning script
"""

import pandas as pd
import pytest
from .clean_dataset import clean_data


@pytest.fixture
def data():
    df = pd.read_csv('../../data/census.csv')
    df = clean_data(df)
    return df


def test_data_nulls(data):
    assert data.shape == data.dropna().shape


def test_data_none(data):
    assert ' ?' not in data.values

