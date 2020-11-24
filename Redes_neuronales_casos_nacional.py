import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

#Preprocesado de los datos
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
mycol = mydb["Nacional_Media_Movil"]
Fecha =[]
Media = []

for x in mycol.find():
    Fecha.append(x["Fecha"])
    Media.append(float(x["Media"]))

df = pd.DataFrame()
df["Fecha"] = pd.to_datetime(Fecha)
df.index = df["Fecha"]
df["Media"] = Media

#cargar dataset
values = df["Media"]
#Asegurarse que los datos estan en float
values = values.astype("float32")
#normalizar características
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.values.reshape(-1, 1)
scaled = scaler.fit_transform(values)
#Con pasos = 7, toma 7 dias y da resultado a 1(var1(t))
#print(scaled)
reframed = series_to_supervised(scaled, PASOS, 1)
#print (reframed.head())

# Dividir datos para entrenar y para prueba
values = reframed.values
n_train_days = 217 - (30+PASOS)
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
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
plt.show()


ultimosDias = df['2020-09-11':'2020-10-11']

values = ultimosDias["Media"]
values = values.astype('float32')
#print(values)

# normalize features
values=values.values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 7, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
print(reframed.head(200))

values = reframed.values
#print(len(values))

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
#print("-------------------------------------")
#adimen = [x for x in results] 
#print(adimen)   
inverted = scaler.inverse_transform(results)


prediccion = pd.DataFrame(inverted)
prediccion.columns = ['pronostico']
#print(prediccion)
prediccion.plot()
plt.show()
