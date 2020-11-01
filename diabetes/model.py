import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

raw_data=pd.read_csv("diabetes.csv")
raw_data.describe()

raw_data.isnull().sum()

plt.figure(dpi=120,figsize=(20,9))
a=raw_data.corr()
sns.heatmap(a,annot=True,linewidths=0.2) 

colums=list(raw_data.columns)
for i in colums:
    plt.figure(figsize=(2,2),dpi=120)
    sns.barplot(x="Outcome",y=i,data=raw_data)

colums=list(raw_data.columns)
for i in colums:
    print("___________________________________________")
    print(i)
    print(raw_data[i].value_counts())
    print("___________________________________________")

data=raw_data.sample(frac=1).reset_index(drop=True)

x=data.drop("Outcome",axis=1)
y=data["Outcome"]

from sklearn.model_selection import train_test_split as tts

train_x,test_x,train_y,test_y=tts(x,y,random_state=56,stratify=y,test_size=0.30)

from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()

train_x=scaler.fit_transform(train_x)

test_x=scaler.fit_transform(test_x)

#from sklearn.linear_model import LogisticRegression
#lr=LogisticRegression()

#lr.fit(train_x,train_y)

#lr.score(train_x,train_y)

#lr.score(test_x,test_y)

from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier(max_depth=7)

#for i in range(1,50):
 #   rf=RandomForestClassifier(max_depth=i)
  #  print(i)
   # rf.fit(train_x,train_y)
    #print("____________________")
    #print(rf.score(train_x,train_y))
    #print("____________________")
    #print(rf.score(test_x,test_y))
    #print("____________________")

rf.fit(train_x,train_y)

rf.score(train_x,train_y)

rf.score(test_x,test_y)

pickle.dump(rf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[8,2,32,34,6,78,9, 9]]))

