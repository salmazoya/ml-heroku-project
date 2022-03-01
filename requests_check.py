"""
Heroku api test script
"""

import requests

data = {
    "age": 32,
    "workclass": "Private",
    "education": "Some-college",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "hoursPerWeek": 60,
    "nativeCountry": "United-States"
}

r = requests.post('https://fastapi2022.herokuapp.com/', json=data)

assert r.status_code == 200

print('response_code', r.status_code)
print('Response:' , r.json())