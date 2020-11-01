
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'age': 1, 'sex': 2, 'cp': 3, 'trestbps': 4, 'chol': 5, 'fbs': 6, 'restecg': 7
                             'thalach': 8, 'exang': 9, 'oldpeak': 10, 'slope': 11, 'ca': 12})

print(r.json())
