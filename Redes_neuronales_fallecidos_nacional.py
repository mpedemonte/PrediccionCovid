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
    mycol = mydb["Nacional_Fallecidos"]
    Fecha =[]
    Casos = []
    Grupo = []
    f1,f2 ,f3,f4,f5,f6,f7= [],[],[],[],[],[],[]
    c1,c2 ,c3,c4,c5,c6,c7= [],[],[],[],[],[],[]
    nc1,nc2 ,nc3,nc4,nc5,nc6,nc7= [],[],[],[],[],[],[]

    for x in mycol.find():
        Grupo.append(x["Grupo_Etario"])
        Fecha.append(x["Fecha"])
        Casos.append(float(x["Media"]))
    print(len(Grupo))
    for i in range(len(Grupo)):
        if Grupo[i] == "<=39":
            f1.append(Fecha[i])
            c1.append(Casos[i])
        if Grupo[i] == "40-49":
            f2.append(Fecha[i])
            c2.append(Casos[i])
        if Grupo[i] == "50-59":
            f3.append(Fecha[i])
            c3.append(Casos[i])
        if Grupo[i] == "60-69":
            f4.append(Fecha[i])
            c4.append(Casos[i])
        if Grupo[i] == "70-79":
            f5.append(Fecha[i])
            c5.append(Casos[i])
        if Grupo[i] == "80-89":
            f6.append(Fecha[i])
            c6.append(Casos[i])
        if Grupo[i] == ">=90":
            f7.append(Fecha[i])
            c7.append(Casos[i])
    """
    for n in range(1,8):
        print(n)
        for i in range(len(c1)):
            if i == 0:
                exec("nc%s.append(c%s[i])" % (n,n))
            else:
                exec("nc%s.append(c%s[i]-c%s[i-1])" %(n,n,n))
    for i in range(1,8):
        exec ("df%s = pd.DataFrame()" %(i))
        exec ("df%s['Fecha'] = pd.to_datetime(f%s)" % (i,i))
        exec ("df%s.index = df%s['Fecha']" % (i,i))
        exec ("df%s['Casos'] = nc%s" % (i,i))"""

    if opcion == 1:
        for i in range(len(c1)):
            if i == 0:
                nc1.append(c1[i])        
            else:
                nc1.append(c1[i]-c1[i-1])
        df1 = pd.DataFrame()
        df1['Fecha'] = pd.to_datetime(f1)
        df1.index = df1['Fecha']
        df1['Casos'] = nc1
        n=df1
    if opcion == 2:
        for i in range(len(c2)):
            if i == 0:
                nc2.append(c2[i])        
            else:
                nc2.append(c2[i]-c2[i-1])
        df2 = pd.DataFrame()
        df2['Fecha'] = pd.to_datetime(f2)
        df2.index = df2['Fecha']
        df2['Casos'] = nc2
        n=df2
    if opcion == 3:
        for i in range(len(c3)):
            if i == 0:
                nc3.append(c3[i])
            else:
                nc3.append(c3[i]-c3[i-1])
        df3 = pd.DataFrame()
        df3['Fecha'] = pd.to_datetime(f3)
        df3.index = df3['Fecha']
        df3['Casos'] = nc3
        n=df3
    if opcion == 4:
        for i in range(len(c4)):
            if i == 0:
                nc4.append(c4[i])
            else:
                nc4.append(c4[i]-c4[i-1])
        df4 = pd.DataFrame()
        df4['Fecha'] = pd.to_datetime(f4)
        df4.index = df4['Fecha']
        df4['Casos'] = nc4
        n=df4
    if opcion == 5:
        for i in range(len(c5)):
            if i == 0:
                nc5.append(c5[i])
            else:
                nc5.append(c5[i]-c5[i-1])
        df5 = pd.DataFrame()
        df5['Fecha'] = pd.to_datetime(f5)
        df5.index = df5['Fecha']
        df5['Casos'] = nc5
        n=df5
    if opcion == 6:
        for i in range(len(c6)):
            if i == 0:
                nc6.append(c6[i])
            else:
                nc6.append(c6[i]-c6[i-1])
        df6 = pd.DataFrame()
        df6['Fecha'] = pd.to_datetime(f6)
        df6.index = df6['Fecha']
        df6['Casos'] = nc6
        n=df6
    if opcion == 7:
        for i in range(len(c7)):
            if i == 0:
                nc7.append(c7[i])
            else:
                nc7.append(c7[i]-c7[i-1])
        df7 = pd.DataFrame()
        df7['Fecha'] = pd.to_datetime(f7)
        df7.index = df7['Fecha']
        df7['Casos'] = nc7
        n=df7

    values = n["Casos"]
    values = values.astype("int32")
    print(values)


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


    mes = n['2020-09-11':'2020-10-11']

    values = mes["Casos"]
    values = values.astype('int32')


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
    #prediccion.plot()
    #plt.show()
    return prediccion
