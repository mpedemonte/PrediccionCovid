import pymongo 
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["PrediccionCovid"]
mycol = mydb["Nacional_Media_Movil"]
Dict = {}
Fecha =[]
Media = []
#array = list(mycol.find())

#print (array)

for x in mycol.find():
    Fecha.append(x["Fecha"])
    Media.append(float(x["Media"]))
#print (Fecha)
#print (Media)
Dict = {"Fecha" : Fecha , "Media" : Media}
#print (Dict)
df = pd.DataFrame(Dict)
#print (df.describe())

X = np.array(df["Media"])
Y = np.array(df["Fecha"])
#print (X.shape)

model = linear_model.LogisticRegression()
model.fit(X,Y)

#print (model)
predictions = model.predict(X)
print(predictions)

