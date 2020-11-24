import pandas as pd
import matplotlib.pyplot as plt
import pymongo 
import os.path
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing

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
for i in Media:
    print(type(i))
df = pd.DataFrame(Dict)
datos = df["Media"]
datos.astype(dtype="float")
print (datos)

modelo = SimpleExpSmoothing(datos).fit(smoothing_level=0.2,optimized=False)
resultado = modelo.fittedvalues

df["Pronostico"] = resultado
print(df)

