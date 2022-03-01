import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from .fast_app import app
import pandas as pd


@pytest.fixture
def client():
    """setting up the app"""
    api_client = TestClient(app)
    return api_client

def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message':'Welcome!'}

def test_get_negative(client):
    r = client.get('/home')
    assert r.status_code != 200

def test_post_1(client):
    r = client.post('/',json={
        "age": 49,
        "workclass": "Self-emp-inc",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "white",
        "sex": "Male",
        "hoursPerWeek": 50,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {'prediction': ' >50k'}

def test_post_2(client):
    r = client.post('/',json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {'prediction': ' <=50k'}

def test_post_missing_data(client):
    r = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "ERROR",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 422