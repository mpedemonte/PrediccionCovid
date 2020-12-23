import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

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

def busca(opcion):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["PrediccionCovid"]
    mycol = mydb["Region_Casos_Sintomaticos"]
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
    
    #print(len(f1))

    """for i in range(1,17):
        exec ("df%s = pd.DataFrame()" %(i))
        exec ("df%s['Fecha'] = pd.to_datetime(f%s)" % (i,i))
        exec ("df%s.index = df%s['Fecha']" % (i,i))
        exec ("df%s['Casos'] = c%s" % (i,i))"""

    if opcion == 1:
        df1 = pd.DataFrame()
        df1['Fecha'] = pd.to_datetime(f1)
        df1.index = df1['Fecha']
        df1['Casos'] = c1
        n=df1

    if opcion == 2:
        df2 = pd.DataFrame()
        df2['Fecha'] = pd.to_datetime(f2)
        df2.index = df2['Fecha']
        df2['Casos'] = c2
        n=df2

    if opcion == 3:
        df3 = pd.DataFrame()
        df3['Fecha'] = pd.to_datetime(f3)
        df3.index = df3['Fecha']
        df3['Casos'] = c3
        n=df3

    if opcion == 4:
        df4 = pd.DataFrame()
        df4['Fecha'] = pd.to_datetime(f4)
        df4.index = df4['Fecha']
        df4['Casos'] = c4
        n=df4

    if opcion == 5:
        df5 = pd.DataFrame()
        df5['Fecha'] = pd.to_datetime(f5)
        df5.index = df5['Fecha']
        df5['Casos'] = c5
        n=df5

    if opcion == 6:
        df6 = pd.DataFrame()
        df6['Fecha'] = pd.to_datetime(f6)
        df6.index = df6['Fecha']
        df6['Casos'] = c6
        n=df6

    if opcion == 7:
        df7 = pd.DataFrame()
        df7['Fecha'] = pd.to_datetime(f7)
        df7.index = df7['Fecha']
        df7['Casos'] = c7
        n=df7

    if opcion == 8:
        df8 = pd.DataFrame()
        df8['Fecha'] = pd.to_datetime(f8)
        df8.index = df8['Fecha']
        df8['Casos'] = c8
        n=df8

    if opcion == 9:
        df9 = pd.DataFrame()
        df9['Fecha'] = pd.to_datetime(f9)
        df9.index = df9['Fecha']
        df9['Casos'] = c9
        n=df9

    if opcion == 10:
        df10 = pd.DataFrame()
        df10['Fecha'] = pd.to_datetime(f10)
        df10.index = df10['Fecha']
        df10['Casos'] = c10
        n=df10

    if opcion == 11:
        df11 = pd.DataFrame()
        df11['Fecha'] = pd.to_datetime(f11)
        df11.index = df11['Fecha']
        df11['Casos'] = c11
        n=df11

    if opcion == 12:
        df12 = pd.DataFrame()
        df12['Fecha'] = pd.to_datetime(f12)
        df12.index = df12['Fecha']
        df12['Casos'] = c12
        n=df12

    if opcion == 13:
        df13 = pd.DataFrame()
        df13['Fecha'] = pd.to_datetime(f13)
        df13.index = df13['Fecha']
        df13['Casos'] = c13
        n=df13

    if opcion == 14:
        df14 = pd.DataFrame()
        df14['Fecha'] = pd.to_datetime(f14)
        df14.index = df14['Fecha']
        df14['Casos'] = c14
        n=df14

    if opcion == 15:
        df15 = pd.DataFrame()
        df15['Fecha'] = pd.to_datetime(f15)
        df15.index = df15['Fecha']
        df15['Casos'] = c15
        n=df15

    if opcion == 16:
        df16 = pd.DataFrame()
        df16['Fecha'] = pd.to_datetime(f16)
        df16.index = df16['Fecha']
        df16['Casos'] = c16
        n=df16

    values = n["Casos"]
    print(values)
    values = values.astype("float32")

    #normalizar características
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values=values.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
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

    results=model.predict(x_val)
    #plt.scatter(range(len(y_val)),y_val,c='g')
    #plt.scatter(range(len(results)),results,c='r')
    #plt.title('validate')
    #plt.show()

    mes = n['2020-09-11':'2020-10-11']
    values = mes["Casos"]
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

    for i in range(len(datos_invertidos)):
        datos_invertidos[i] = int(datos_invertidos[i])


    prediccion = pd.DataFrame(datos_invertidos)
    prediccion.columns = ['pronostico']
    return prediccion
    #prediccion.plot()
    #plt.show()