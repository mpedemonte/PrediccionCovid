import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

PASOS=7

# convertir series en aprendizaje supervisado
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model

def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["PrediccionCovid"]
mycol = mydb["Region_Casos_Asintomaticos"]
Region = []
Fecha =[]
Casos = []
f1,f2 ,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
c1,c2 ,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
for x in mycol.find():
    Region.append(x["Region"])
    Fecha.append(x["Fecha"])
    Casos.append(float(x["Casos"]))

for i in range(len(Region)):
    if Region[i] == "Arica y Parinacota":
        f1.append(Fecha[i])
        c1.append(Casos[i])
    if Region[i] == "Tarapacá":
        f2.append(Fecha[i])
        c2.append(Casos[i])
    if Region[i] == "Antofagasta":
        f3.append(Fecha[i])
        c3.append(Casos[i])
    if Region[i] == "Atacama":
        f4.append(Fecha[i])
        c4.append(Casos[i])
    if Region[i] == "Coquimbo":
        f5.append(Fecha[i])
        c5.append(Casos[i])
    if Region[i] == "Valparaíso":
        f6.append(Fecha[i])
        c6.append(Casos[i])
    if Region[i] == "Metropolitana":
        f7.append(Fecha[i])
        c7.append(Casos[i])
    if Region[i] == "O’Higgins":
        f8.append(Fecha[i])
        c8.append(Casos[i])
    if Region[i] == "Maule":
        f9.append(Fecha[i])
        c9.append(Casos[i])
    if Region[i] == "Ñuble":
        f10.append(Fecha[i])
        c10.append(Casos[i])
    if Region[i] == "Biobío":
        f11.append(Fecha[i])
        c11.append(Casos[i])
    if Region[i] == "Araucanía":
        f12.append(Fecha[i])
        c12.append(Casos[i])
    if Region[i] == "Los Ríos":
        f13.append(Fecha[i])
        c13.append(Casos[i])
    if Region[i] == "Los Lagos":
        f14.append(Fecha[i])
        c14.append(Casos[i])
    if Region[i] == "Aysén":
        f15.append(Fecha[i])
        c15.append(Casos[i])
    if Region[i] == "Magallanes":
        f16.append(Fecha[i])
        c16.append(Casos[i])
  
print(len(f1))

for i in range(1,17):
     exec ("df%s = pd.DataFrame()" %(i))
     exec ("df%s['Fecha'] = pd.to_datetime(f%s)" % (i,i))
     exec ("df%s.index = df%s['Fecha']" % (i,i))
     exec ("df%s['Casos'] = c%s" % (i,i))
n=df7
values = n["Casos"]
print(values)
values = values.astype("float32")

scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.values.reshape(-1, 1)
#print (values)
scaled = scaler.fit_transform(values)

# Dividir datos para entrenar y para prueba
reframed = series_to_supervised(scaled, PASOS, 1)
values = reframed.values
n_train_days = 166 - (30+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# dividir en entradas y salidas
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

EPOCHS=300

model = crear_modeloFF()
history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)

results=model.predict(x_val)
#plt.scatter(range(len(y_val)),y_val,c='g')
#plt.scatter(range(len(results)),results,c='r')
#plt.title('validate')
#plt.show()

ultimosDias = df7['2020-09-11':'2020-10-11']
values = ultimosDias["Casos"]
values = values.astype('float32')

# normalize features
values=values.values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 7, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
print(reframed.head(200))

values = reframed.values
print(len(values))

x_test = values[len(values)-1:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
#print(x_test)

results=[]
for i in range(30):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    #print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])
print(results)
#adimen = [x for x in results] 
#print(adimen)   
inverted = scaler.inverse_transform(results)

for i in range(len(inverted)):
    inverted[i] = int(inverted[i])

prediccion = pd.DataFrame(inverted)
prediccion.columns = ['pronostico']
#print(prediccion)
prediccion.plot()
plt.show()