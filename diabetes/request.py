
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={ 'Pregnancies':8, 'Glucose':2 , 'BloodPressure':32 , 'SkinThickness':34 , 'Insulin':6, 'BMI': 78, 'DiabetesPedigreeFunction': 9
                             'Age':9})

print(r.json())
