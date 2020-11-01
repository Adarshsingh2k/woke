from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import pickle
raw_data = pd.read_csv('data.csv')
raw_data.isnull().sum()

i = list(raw_data.columns)

for l in i:
    print('---------------------------')
    print(l)
    print(raw_data[l].value_counts())
    print('')

raw_data['trestbps'].replace({'?': 120, '120': 120}, inplace=True)
raw_data['fbs'].replace({'?': 0, '0': 0}, inplace=True)
raw_data['restecg'].replace({'?': 0, '0': 0}, inplace=True)
raw_data['exang'].replace({'?': 0, '0': 0}, inplace=True)
raw_data['thalach'].replace({'?': 150, '150': 150}, inplace=True)


raw_data.head()

raw_data['chol'].replace({'?': 130}, inplace=True)

raw_data['slope'].replace({'?': 2, '2.00': 2}, inplace=True)

raw_data['ca'].replace({'?': 1}, inplace=True)

raw_data['target'].value_counts()

new_data = raw_data.drop('thal', axis=1)
new_data.head()

data = new_data.sample(frac=1).reset_index(drop=True)

x = data.drop('target', axis=1)
y = data['target'].values.reshape(-1, 1)


x.shape, y.shape

train_x, test_x, train_y, test_y = tts(x, y, random_state=56, test_size=.30)


scaler = StandardScaler()
x = scaler.fit_transform(x)


#train_x.shape
#train_y.shape
#test_x.shape
#test_y.shape


train_x.shape
test_x.shape


logreg = LogisticRegression()
logreg.fit(train_x, train_y.ravel())

logreg.score(train_x, train_y)

Pred = logreg.predict(test_x)
logreg.score(test_x, test_y)

logreg.score(test_x, test_y)

rf = RandomForestClassifier()

rf.fit(train_x, train_y)


rf.score(train_x, train_y), rf.score(test_x, test_y)

pickle.dump(rf, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]))
