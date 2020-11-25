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
def Series_a_AprendizajeSupervizado(data, n_in=1, n_out=1, dropnan=True):
    if type(data) is list :
        n_vars = 1
    else: 
        n_vars = data.shape[1]
        print(data.shape[1])
    df = pd.DataFrame(data)
    cols, nombres = list(), list()
    # secuencia de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        nombres += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # secuencia de pronóstico (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            nombres += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            nombres += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # colocar todo junto
    agg = pd.concat(cols, axis=1)
    agg.columns = nombres
    # borrar filas con valores NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def crear_modeloFF():
    model = Sequential() #Crea una serie de capas de neuronas sequencialmente
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh')) #capa de entrada = PASOS, input_shape(cant_capasOcultas, cant_neuronas)   Funcion tangente hiperbolica
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"]) #Con esto indicamos el tipo de pérdida (loss) que utilizaremos, 
                                                                                #el “optimizador” de los pesos de las conexiones de las neuronas y las métricas que queremos obtener.
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
print(type(scaled))

#Con pasos = 7, toma 7 dias y da resultado a 1(var1(t))

reframed = Series_a_AprendizajeSupervizado(scaled, PASOS, 1)
print (reframed)
# Dividir datos para entrenar y para prueba
values = reframed.values
print(len(values))
n_dias_entrenamiento = len(values) - (30)
entrenamiento = values[:n_dias_entrenamiento, :]
prueba = values[n_dias_entrenamiento:, :]
# dividir en entradas y salidas
x_entrenamiento, y_entrenamiento = entrenamiento[:, :-1], entrenamiento[:, -1]
x_val, y_val = prueba[:, :-1], prueba[:, -1]
# remodelar la entrada para que sea 3D [muestras, pasos de tiempo, características]

x_entrenamiento = x_entrenamiento.reshape((x_entrenamiento.shape[0], 1, x_entrenamiento.shape[1]))

x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#print(x_entrenamiento.shape, y_entrenamiento.shape, x_val.shape, y_val.shape)

EPOCHS=40

model = crear_modeloFF()
history=model.fit(x_entrenamiento,y_entrenamiento,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)

resultados=model.predict(x_val)
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(resultados)),resultados,c='r')
plt.title('validate')
#plt.show()


mes = df['2020-09-11':'2020-10-11']

values = mes["Media"]
values = values.astype('float32')


# normalizar características
values=values.values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = Series_a_AprendizajeSupervizado(scaled, 7, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
print(reframed)

values = reframed.values
x_test = values[len(values)-1:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

resultados=[]
for i in range(30):
    parcial=model.predict(x_test)
    print(parcial[0])
    resultados.append(parcial[0])
    x_test=agregarNuevoValor(x_test,parcial[0])
print(resultados)
 
datos_invertidos = scaler.inverse_transform(resultados)


prediccion = pd.DataFrame(datos_invertidos)
prediccion.columns = ['pronostico']
prediccion.plot()
plt.show()
