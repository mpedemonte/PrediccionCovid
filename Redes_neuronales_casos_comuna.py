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


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["PrediccionCovid"]
mycol = mydb["Comuna_Casos"]
Region = []
Fecha =[]
Casos = []
Comuna = []
for i in range(1,362):
     exec ("f%s = []" %(i))
     exec ("c%s = []" %(i))
     exec ("nc%s = []" %(i))

for x in mycol.find():
    Region.append(x["Region"])
    Fecha.append(x["Fecha"])
    Casos.append(float(x["Casos"]))
    Comuna.append(x["Comuna"])

for i in range(len(Region)):
    if Region[i] == "Arica y Parinacota":
        if Comuna[i]=="Arica":
            f1.append(Fecha[i])
            c1.append(Casos[i])
        if Comuna[i]=="Camarones":
            f2.append(Fecha[i])
            c2.append(Casos[i])
        if Comuna[i]=="General Lagos":
            f3.append(Fecha[i])
            c3.append(Casos[i])
        if Comuna[i]=="Putre":
            f4.append(Fecha[i])
            c4.append(Casos[i])
        if Comuna[i]=="Desconocido Arica y Parinacota":
            f5.append(Fecha[i])
            c5.append(Casos[i])
    if Region[i] == "Tarapacá":
        if Comuna[i]=="Alto Hospicio":
            f6.append(Fecha[i])
            c6.append(Casos[i])
        if Comuna[i]=="Camina":
            f7.append(Fecha[i])
            c7.append(Casos[i])
        if Comuna[i]=="Colchane":
            f8.append(Fecha[i])
            c8.append(Casos[i])
        if Comuna[i]=="Huara":
            f9.append(Fecha[i])
            c9.append(Casos[i])
        if Comuna[i]=="Iquique":
            f10.append(Fecha[i])
            c10.append(Casos[i])
        if Comuna[i]=="Pica":
            f11.append(Fecha[i])
            c11.append(Casos[i])
        if Comuna[i]=="Pozo Almonte":
            f12.append(Fecha[i])
            c12.append(Casos[i])
        if Comuna[i]=="Desconocido Tarapaca":
            f13.append(Fecha[i])
            c13.append(Casos[i])
    if Region[i] == "Antofagasta":
        if Comuna[i]=="Antofagasta":
            f14.append(Fecha[i])
            c14.append(Casos[i])
        if Comuna[i]=="Calama":
            f15.append(Fecha[i])
            c15.append(Casos[i])
        if Comuna[i]=="Maria Elena":
            f16.append(Fecha[i])
            c16.append(Casos[i])
        if Comuna[i]=="Mejillones":
            f17.append(Fecha[i])
            c17.append(Casos[i])
        if Comuna[i]=="Ollague":
            f18.append(Fecha[i])
            c18.append(Casos[i])
        if Comuna[i]=="San Pedro de Atacama":
            f19.append(Fecha[i])
            c19.append(Casos[i])
        if Comuna[i]=="Sierra Gorda":
            f20.append(Fecha[i])
            c20.append(Casos[i])
        if Comuna[i]=="Taltal":
            f21.append(Fecha[i])
            c21.append(Casos[i])
        if Comuna[i]=="Tocopilla":
            f22.append(Fecha[i])
            c22.append(Casos[i])
        if Comuna[i]=="Desconocido Antofagasta":
            f23.append(Fecha[i])
            c23.append(Casos[i])
    if Region[i] == "Atacama":
        if Comuna[i]=="Alto del Carmen":
            f24.append(Fecha[i])
            c24.append(Casos[i])
        if Comuna[i]=="Caldera":
            f25.append(Fecha[i])
            c25.append(Casos[i])
        if Comuna[i]=="Chanaral":
            f26.append(Fecha[i])
            c26.append(Casos[i])
        if Comuna[i]=="Copiapo":
            f27.append(Fecha[i])
            c27.append(Casos[i])
        if Comuna[i]=="Diego de Almagro":
            f28.append(Fecha[i])
            c28.append(Casos[i])
        if Comuna[i]=="Freirina":
            f29.append(Fecha[i])
            c29.append(Casos[i])
        if Comuna[i]=="Huasco":
            f30.append(Fecha[i])
            c30.append(Casos[i])
        if Comuna[i]=="Tierra Amarilla":
            f31.append(Fecha[i])
            c31.append(Casos[i])
        if Comuna[i]=="Vallena":
            f32.append(Fecha[i])
            c32.append(Casos[i])
        if Comuna[i]=="Desconocido Atacama":
            f33.append(Fecha[i])
            c33.append(Casos[i])
    if Region[i] == "Coquimbo":
        if Comuna[i]=="Andacollo":
            f34.append(Fecha[i])
            c34.append(Casos[i])
        if Comuna[i]=="Canela":
            f35.append(Fecha[i])
            c35.append(Casos[i])
        if Comuna[i]=="Combarbala":
            f36.append(Fecha[i])
            c36.append(Casos[i])
        if Comuna[i]=="Coquimbo":
            f37.append(Fecha[i])
            c37.append(Casos[i])
        if Comuna[i]=="Illapel":
            f38.append(Fecha[i])
            c38.append(Casos[i])
        if Comuna[i]=="La Higuera":
            f39.append(Fecha[i])
            c39.append(Casos[i])
        if Comuna[i]=="La Serena":
            f40.append(Fecha[i])
            c40.append(Casos[i])
        if Comuna[i]=="Los Vilos":
            f41.append(Fecha[i])
            c41.append(Casos[i])
        if Comuna[i]=="Monte Patria":
            f42.append(Fecha[i])
            c42.append(Casos[i])
        if Comuna[i]=="Ovalle":
            f43.append(Fecha[i])
            c43.append(Casos[i])
        if Comuna[i]=="Paiguano":
            f44.append(Fecha[i])
            c44.append(Casos[i])
        if Comuna[i]=="Punitaqui":
            f45.append(Fecha[i])
            c45.append(Casos[i])
        if Comuna[i]=="Rio Hurtado":
            f46.append(Fecha[i])
            c46.append(Casos[i])
        if Comuna[i]=="Salamanca":
            f47.append(Fecha[i])
            c47.append(Casos[i])
        if Comuna[i]=="Vicuna":
            f48.append(Fecha[i])
            c48.append(Casos[i])
        if Comuna[i]=="Desconocido Coquimbo":
            f49.append(Fecha[i])
            c49.append(Casos[i])
    if Region[i] == "Valparaíso":
        if Comuna[i]=="Algarrobo":
            f50.append(Fecha[i])
            c50.append(Casos[i])
        if Comuna[i]=="Cabildo":
            f51.append(Fecha[i])
            c51.append(Casos[i])
        if Comuna[i]=="Calera":
            f52.append(Fecha[i])
            c52.append(Casos[i])
        if Comuna[i]=="Calle Larga":
            f53.append(Fecha[i])
            c53.append(Casos[i])
        if Comuna[i]=="Cartagena":
            f54.append(Fecha[i])
            c54.append(Casos[i])
        if Comuna[i]=="Casablanca":
            f55.append(Fecha[i])
            c55.append(Casos[i])
        if Comuna[i]=="Catemu":
            f56.append(Fecha[i])
            c56.append(Casos[i])
        if Comuna[i]=="Concon":
            f57.append(Fecha[i])
            c57.append(Casos[i])
        if Comuna[i]=="El Quisco":
            f58.append(Fecha[i])
            c58.append(Casos[i])
        if Comuna[i]=="El Tabo":
            f59.append(Fecha[i])
            c59.append(Casos[i])
        if Comuna[i]=="Hijuelas":
            f60.append(Fecha[i])
            c60.append(Casos[i])
        if Comuna[i]=="Isla de Pascua":
            f61.append(Fecha[i])
            c61.append(Casos[i])
        if Comuna[i]=="Juan Fernandez":
            f62.append(Fecha[i])
            c62.append(Casos[i])
        if Comuna[i]=="La Cruz":
            f63.append(Fecha[i])
            c63.append(Casos[i])
        if Comuna[i]=="La Ligua":
            f64.append(Fecha[i])
            c64.append(Casos[i])
        if Comuna[i]=="Limache":
            f65.append(Fecha[i])
            c65.append(Casos[i])
        if Comuna[i]=="Llaillay":
            f66.append(Fecha[i])
            c66.append(Casos[i])
        if Comuna[i]=="Los Andes":
            f67.append(Fecha[i])
            c67.append(Casos[i])
        if Comuna[i]=="Nogales":
            f68.append(Fecha[i])
            c68.append(Casos[i])
        if Comuna[i]=="Olmue":
            f69.append(Fecha[i])
            c69.append(Casos[i])
        if Comuna[i]=="Panquehue":
            f70.append(Fecha[i])
            c70.append(Casos[i])
        if Comuna[i]=="Papudo":
            f71.append(Fecha[i])
            c71.append(Casos[i])
        if Comuna[i]=="Petorca":
            f72.append(Fecha[i])
            c72.append(Casos[i])
        if Comuna[i]=="Puchuncavi":
            f73.append(Fecha[i])
            c73.append(Casos[i])
        if Comuna[i]=="Putaendo":
            f74.append(Fecha[i])
            c74.append(Casos[i])
        if Comuna[i]=="Quillota":
            f75.append(Fecha[i])
            c75.append(Casos[i])
        if Comuna[i]=="Quilpue":
            f76.append(Fecha[i])
            c76.append(Casos[i])
        if Comuna[i]=="Quintero":
            f77.append(Fecha[i])
            c77.append(Casos[i])
        if Comuna[i]=="Rinconada":
            f78.append(Fecha[i])
            c78.append(Casos[i])
        if Comuna[i]=="San Antonio":
            f79.append(Fecha[i])
            c79.append(Casos[i])
        if Comuna[i]=="San Esteban":
            f80.append(Fecha[i])
            c80.append(Casos[i])
        if Comuna[i]=="San Felipe":
            f81.append(Fecha[i])
            c81.append(Casos[i])
        if Comuna[i]=="Santa Maria":
            f82.append(Fecha[i])
            c82.append(Casos[i])
        if Comuna[i]=="Santo Domingo":
            f83.append(Fecha[i])
            c83.append(Casos[i])
        if Comuna[i]=="Valparaiso":
            f84.append(Fecha[i])
            c84.append(Casos[i])
        if Comuna[i]=="Villa Alemana":
            f85.append(Fecha[i])
            c85.append(Casos[i])
        if Comuna[i]=="Vina del Mar":
            f86.append(Fecha[i])
            c86.append(Casos[i])
        if Comuna[i]=="Zapallar":
            f87.append(Fecha[i])
            c87.append(Casos[i])
        if Comuna[i]=="Desconocido Valparaiso":
            f88.append(Fecha[i])
            c88.append(Casos[i])
    if Region[i] == "Metropolitana":
        if Comuna[i]=="Alhue":
            f89.append(Fecha[i])
            c89.append(Casos[i])
        if Comuna[i]=="Buin":
            f90.append(Fecha[i])
            c90.append(Casos[i])
        if Comuna[i]=="Calera de Tango":
            f91.append(Fecha[i])
            c91.append(Casos[i])
        if Comuna[i]=="Cerrillos":
            f92.append(Fecha[i])
            c92.append(Casos[i])
        if Comuna[i]=="Cerro Navia":
            f93.append(Fecha[i])
            c93.append(Casos[i])
        if Comuna[i]=="Colina":
            f94.append(Fecha[i])
            c94.append(Casos[i])
        if Comuna[i]=="Conchali":
            f95.append(Fecha[i])
            c95.append(Casos[i])
        if Comuna[i]=="Curacavi":
            f96.append(Fecha[i])
            c96.append(Casos[i])
        if Comuna[i]=="El Bosque":
            f97.append(Fecha[i])
            c97.append(Casos[i])
        if Comuna[i]=="El Monte":
            f98.append(Fecha[i])
            c98.append(Casos[i])
        if Comuna[i]=="Estacion Central":
            f99.append(Fecha[i])
            c99.append(Casos[i])
        if Comuna[i]=="Huechuraba":
            f100.append(Fecha[i])
            c100.append(Casos[i])
        if Comuna[i]=="Independencia":
            f101.append(Fecha[i])
            c101.append(Casos[i])
        if Comuna[i]=="Isla de Maipo":
            f102.append(Fecha[i])
            c102.append(Casos[i])
        if Comuna[i]=="La Cisterna":
            f103.append(Fecha[i])
            c103.append(Casos[i])
        if Comuna[i]=="La Florida":
            f104.append(Fecha[i])
            c104.append(Casos[i])
        if Comuna[i]=="La Granja":
            f105.append(Fecha[i])
            c105.append(Casos[i])
        if Comuna[i]=="La Pintana":
            f106.append(Fecha[i])
            c106.append(Casos[i])
        if Comuna[i]=="La Reina":
            f107.append(Fecha[i])
            c107.append(Casos[i])
        if Comuna[i]=="Lampa":
            f108.append(Fecha[i])
            c108.append(Casos[i])
        if Comuna[i]=="Las Condes":
            f109.append(Fecha[i])
            c109.append(Casos[i])
        if Comuna[i]=="Lo Barnechea":
            f110.append(Fecha[i])
            c110.append(Casos[i])
        if Comuna[i]=="Lo Espejo":
            f111.append(Fecha[i])
            c111.append(Casos[i])
        if Comuna[i]=="Lo Prado":
            f112.append(Fecha[i])
            c112.append(Casos[i])
        if Comuna[i]=="Macul":
            f113.append(Fecha[i])
            c113.append(Casos[i])
        if Comuna[i]=="Maipu":
            f114.append(Fecha[i])
            c114.append(Casos[i])
        if Comuna[i]=="Maria Pinto":
            f115.append(Fecha[i])
            c115.append(Casos[i])
        if Comuna[i]=="Melipilla":
            f116.append(Fecha[i])
            c116.append(Casos[i])
        if Comuna[i]=="Nunoa":
            f117.append(Fecha[i])
            c117.append(Casos[i])
        if Comuna[i]=="Padre Hurtado":
            f118.append(Fecha[i])
            c118.append(Casos[i])
        if Comuna[i]=="Paine":
            f119.append(Fecha[i])
            c119.append(Casos[i])
        if Comuna[i]=="Pedro Aguirre Cerda":
            f120.append(Fecha[i])
            c120.append(Casos[i])
        if Comuna[i]=="Penaflor":
            f121.append(Fecha[i])
            c121.append(Casos[i])
        if Comuna[i]=="Penalolen":
            f122.append(Fecha[i])
            c122.append(Casos[i])
        if Comuna[i]=="Pirque":
            f123.append(Fecha[i])
            c123.append(Casos[i])
        if Comuna[i]=="Providencia":
            f124.append(Fecha[i])
            c124.append(Casos[i])
        if Comuna[i]=="Pudahuel":
            f125.append(Fecha[i])
            c125.append(Casos[i])
        if Comuna[i]=="Puente Alto":
            f126.append(Fecha[i])
            c126.append(Casos[i])
        if Comuna[i]=="Quilicura":
            f127.append(Fecha[i])
            c127.append(Casos[i])
        if Comuna[i]=="Quinta Normal":
            f128.append(Fecha[i])
            c128.append(Casos[i])
        if Comuna[i]=="Recoleta":
            f129.append(Fecha[i])
            c129.append(Casos[i])
        if Comuna[i]=="Renca":
            f130.append(Fecha[i])
            c130.append(Casos[i])
        if Comuna[i]=="San Bernardo":
            f131.append(Fecha[i])
            c131.append(Casos[i])
        if Comuna[i]=="San Joaquin":
            f132.append(Fecha[i])
            c132.append(Casos[i])
        if Comuna[i]=="San Jose de Maipo":
            f133.append(Fecha[i])
            c133.append(Casos[i])
        if Comuna[i]=="San Miguel":
            f134.append(Fecha[i])
            c134.append(Casos[i])
        if Comuna[i]=="San Pedro":
            f135.append(Fecha[i])
            c135.append(Casos[i])
        if Comuna[i]=="San Ramon":
            f136.append(Fecha[i])
            c136.append(Casos[i])
        if Comuna[i]=="Santiago":
            f137.append(Fecha[i])
            c137.append(Casos[i])
        if Comuna[i]=="Talagante":
            f138.append(Fecha[i])
            c138.append(Casos[i])
        if Comuna[i]=="Tiltil":
            f139.append(Fecha[i])
            c139.append(Casos[i])
        if Comuna[i]=="Vitacura":
            f140.append(Fecha[i])
            c140.append(Casos[i])
        if Comuna[i]=="Desconocido Metropolitana":
            f141.append(Fecha[i])
            c141.append(Casos[i])
    if Region[i] == "O’Higgins":
        if Comuna[i]=="Chepica":
            f142.append(Fecha[i])
            c142.append(Casos[i])
        if Comuna[i]=="Chimbarongo":
            f143.append(Fecha[i])
            c143.append(Casos[i])
        if Comuna[i]=="Codegua":
            f144.append(Fecha[i])
            c144.append(Casos[i])
        if Comuna[i]=="Coinco":
            f145.append(Fecha[i])
            c145.append(Casos[i])
        if Comuna[i]=="Coltauco":
            f146.append(Fecha[i])
            c146.append(Casos[i])
        if Comuna[i]=="Donihue":
            f147.append(Fecha[i])
            c147.append(Casos[i])
        if Comuna[i]=="Graneros":
            f148.append(Fecha[i])
            c148.append(Casos[i])
        if Comuna[i]=="La Estrella":
            f149.append(Fecha[i])
            c149.append(Casos[i])
        if Comuna[i]=="Las Cabras":
            f150.append(Fecha[i])
            c150.append(Casos[i])
        if Comuna[i]=="Litueche":
            f151.append(Fecha[i])
            c151.append(Casos[i])
        if Comuna[i]=="Lolol":
            f152.append(Fecha[i])
            c152.append(Casos[i])
        if Comuna[i]=="Machali":
            f153.append(Fecha[i])
            c153.append(Casos[i])
        if Comuna[i]=="Malloa":
            f154.append(Fecha[i])
            c154.append(Casos[i])
        if Comuna[i]=="Marchihue":
            f155.append(Fecha[i])
            c155.append(Casos[i])
        if Comuna[i]=="Mostazal":
            f156.append(Fecha[i])
            c156.append(Casos[i])
        if Comuna[i]=="Nancagua":
            f157.append(Fecha[i])
            c157.append(Casos[i])
        if Comuna[i]=="Navidad":
            f158.append(Fecha[i])
            c158.append(Casos[i])
        if Comuna[i]=="Olivar":
            f159.append(Fecha[i])
            c159.append(Casos[i])
        if Comuna[i]=="Palmilla":
            f160.append(Fecha[i])
            c160.append(Casos[i])
        if Comuna[i]=="Paredones":
            f161.append(Fecha[i])
            c161.append(Casos[i])
        if Comuna[i]=="Parelillo":
            f162.append(Fecha[i])
            c162.append(Casos[i])
        if Comuna[i]=="Peumo":
            f163.append(Fecha[i])
            c163.append(Casos[i])
        if Comuna[i]=="Pichidegua":
            f164.append(Fecha[i])
            c164.append(Casos[i])
        if Comuna[i]=="Pichilemu":
            f165.append(Fecha[i])
            c165.append(Casos[i])
        if Comuna[i]=="Placilla":
            f166.append(Fecha[i])
            c166.append(Casos[i])
        if Comuna[i]=="Pumanque":
            f167.append(Fecha[i])
            c167.append(Casos[i])
        if Comuna[i]=="Quinta de Tilcoco":
            f168.append(Fecha[i])
            c168.append(Casos[i])
        if Comuna[i]=="Rancagua":
            f169.append(Fecha[i])
            c169.append(Casos[i])
        if Comuna[i]=="Rengo":
            f170.append(Fecha[i])
            c170.append(Casos[i])
        if Comuna[i]=="Requinoa":
            f171.append(Fecha[i])
            c171.append(Casos[i])
        if Comuna[i]=="San Fernando":
            f172.append(Fecha[i])
            c172.append(Casos[i])
        if Comuna[i]=="San Vicente":
            f173.append(Fecha[i])
            c173.append(Casos[i])
        if Comuna[i]=="Santa Cruz":
            f174.append(Fecha[i])
            c174.append(Casos[i])
        if Comuna[i]=="Desconocido O’Higgins":
            f175.append(Fecha[i])
            c175.append(Casos[i])
    if Region[i] == "Maule":
        if Comuna[i]=="Cauquenes":
            f176.append(Fecha[i])
            c176.append(Casos[i])
        if Comuna[i]=="Chanco":
            f177.append(Fecha[i])
            c177.append(Casos[i])
        if Comuna[i]=="Colbun":
            f178.append(Fecha[i])
            c178.append(Casos[i])
        if Comuna[i]=="Constitucion":
            f179.append(Fecha[i])
            c179.append(Casos[i])
        if Comuna[i]=="Curepto":
            f180.append(Fecha[i])
            c180.append(Casos[i])
        if Comuna[i]=="Curico":
            f181.append(Fecha[i])
            c181.append(Casos[i])
        if Comuna[i]=="Empedrado":
            f182.append(Fecha[i])
            c182.append(Casos[i])
        if Comuna[i]=="Hualane":
            f183.append(Fecha[i])
            c183.append(Casos[i])
        if Comuna[i]=="Licanten":
            f184.append(Fecha[i])
            c184.append(Casos[i])
        if Comuna[i]=="Linares":
            f185.append(Fecha[i])
            c185.append(Casos[i])
        if Comuna[i]=="Longavi":
            f186.append(Fecha[i])
            c186.append(Casos[i])
        if Comuna[i]=="Maule":
            f187.append(Fecha[i])
            c187.append(Casos[i])
        if Comuna[i]=="Molina":
            f188.append(Fecha[i])
            c188.append(Casos[i])
        if Comuna[i]=="Parral":
            f189.append(Fecha[i])
            c189.append(Casos[i])
        if Comuna[i]=="Pelarco":
            f190.append(Fecha[i])
            c190.append(Casos[i])
        if Comuna[i]=="Pelluhue":
            f191.append(Fecha[i])
            c191.append(Casos[i])
        if Comuna[i]=="Pencahue":
            f192.append(Fecha[i])
            c192.append(Casos[i])
        if Comuna[i]=="Rauco":
            f193.append(Fecha[i])
            c193.append(Casos[i])
        if Comuna[i]=="Retiro":
            f194.append(Fecha[i])
            c194.append(Casos[i])
        if Comuna[i]=="Rio Claro":
            f195.append(Fecha[i])
            c195.append(Casos[i])
        if Comuna[i]=="Romeral":
            f196.append(Fecha[i])
            c196.append(Casos[i])
        if Comuna[i]=="Sagrada Familia":
            f197.append(Fecha[i])
            c197.append(Casos[i])
        if Comuna[i]=="San Clemente":
            f198.append(Fecha[i])
            c198.append(Casos[i])
        if Comuna[i]=="San Javier":
            f199.append(Fecha[i])
            c199.append(Casos[i])
        if Comuna[i]=="San Rafael":
            f200.append(Fecha[i])
            c200.append(Casos[i])
        if Comuna[i]=="Talca":
            f201.append(Fecha[i])
            c201.append(Casos[i])
        if Comuna[i]=="Teno":
            f202.append(Fecha[i])
            c202.append(Casos[i])
        if Comuna[i]=="Vichuquen":
            f203.append(Fecha[i])
            c203.append(Casos[i])
        if Comuna[i]=="Villa Alegre":
            f204.append(Fecha[i])
            c204.append(Casos[i])
        if Comuna[i]=="Yerbas Buenas":
            f205.append(Fecha[i])
            c205.append(Casos[i])
        if Comuna[i]=="Desconocido Maule":
            f206.append(Fecha[i])
            c206.append(Casos[i])
    if Region[i] == "Ñuble":
        if Comuna[i]=="Bulnes":
            f207.append(Fecha[i])
            c207.append(Casos[i])
        if Comuna[i]=="Chillan":
            f208.append(Fecha[i])
            c208.append(Casos[i])
        if Comuna[i]=="Chillan Viejo":
            f209.append(Fecha[i])
            c209.append(Casos[i])
        if Comuna[i]=="Cobquecura":
            f210.append(Fecha[i])
            c210.append(Casos[i])
        if Comuna[i]=="Coelemu":
            f211.append(Fecha[i])
            c211.append(Casos[i])
        if Comuna[i]=="Coihueco":
            f212.append(Fecha[i])
            c212.append(Casos[i])
        if Comuna[i]=="El Carmen":
            f213.append(Fecha[i])
            c213.append(Casos[i])
        if Comuna[i]=="Ninhue":
            f214.append(Fecha[i])
            c214.append(Casos[i])
        if Comuna[i]=="Niquen":
            f215.append(Fecha[i])
            c215.append(Casos[i])
        if Comuna[i]=="Pemuco":
            f216.append(Fecha[i])
            c216.append(Casos[i])
        if Comuna[i]=="Pinto":
            f217.append(Fecha[i])
            c217.append(Casos[i])
        if Comuna[i]=="Portezuelo":
            f218.append(Fecha[i])
            c218.append(Casos[i])
        if Comuna[i]=="Quillon":
            f219.append(Fecha[i])
            c219.append(Casos[i])
        if Comuna[i]=="Quirihue":
            f220.append(Fecha[i])
            c220.append(Casos[i])
        if Comuna[i]=="Ranquil":
            f221.append(Fecha[i])
            c221.append(Casos[i])
        if Comuna[i]=="San Carlos":
            f222.append(Fecha[i])
            c222.append(Casos[i])
        if Comuna[i]=="San Fabian":
            f223.append(Fecha[i])
            c223.append(Casos[i])
        if Comuna[i]=="San Ignacio":
            f224.append(Fecha[i])
            c224.append(Casos[i])
        if Comuna[i]=="San Nicolas":
            f225.append(Fecha[i])
            c225.append(Casos[i])
        if Comuna[i]=="Treguaco":
            f226.append(Fecha[i])
            c226.append(Casos[i])
        if Comuna[i]=="Yungay":
            f227.append(Fecha[i])
            c227.append(Casos[i])
        if Comuna[i]=="Desconocido Nuble":
            f228.append(Fecha[i])
            c228.append(Casos[i])
    if Region[i] == "Biobío":
        if Comuna[i]=="Alto Biobio":
            f229.append(Fecha[i])
            c229.append(Casos[i])
        if Comuna[i]=="Antuco":
            f230.append(Fecha[i])
            c230.append(Casos[i])
        if Comuna[i]=="Arauco":
            f231.append(Fecha[i])
            c231.append(Casos[i])
        if Comuna[i]=="Cabrero":
            f232.append(Fecha[i])
            c232.append(Casos[i])
        if Comuna[i]=="Canete":
            f233.append(Fecha[i])
            c233.append(Casos[i])
        if Comuna[i]=="Chiguayante":
            f234.append(Fecha[i])
            c234.append(Casos[i])
        if Comuna[i]=="Concepcion":
            f235.append(Fecha[i])
            c235.append(Casos[i])
        if Comuna[i]=="Contulmo":
            f236.append(Fecha[i])
            c236.append(Casos[i])
        if Comuna[i]=="Coronel":
            f237.append(Fecha[i])
            c237.append(Casos[i])
        if Comuna[i]=="Curanilahue":
            f238.append(Fecha[i])
            c238.append(Casos[i])
        if Comuna[i]=="Florida":
            f239.append(Fecha[i])
            c239.append(Casos[i])
        if Comuna[i]=="Hualpen":
            f240.append(Fecha[i])
            c240.append(Casos[i])
        if Comuna[i]=="Hualqui":
            f241.append(Fecha[i])
            c241.append(Casos[i])
        if Comuna[i]=="Laja":
            f242.append(Fecha[i])
            c242.append(Casos[i])
        if Comuna[i]=="Lebu":
            f243.append(Fecha[i])
            c243.append(Casos[i])
        if Comuna[i]=="Los Alamo":
            f244.append(Fecha[i])
            c244.append(Casos[i])
        if Comuna[i]=="Los Angeles":
            f245.append(Fecha[i])
            c245.append(Casos[i])
        if Comuna[i]=="Lota":
            f246.append(Fecha[i])
            c246.append(Casos[i])
        if Comuna[i]=="Mulchen":
            f247.append(Fecha[i])
            c247.append(Casos[i])
        if Comuna[i]=="Nacimiento":
            f248.append(Fecha[i])
            c248.append(Casos[i])
        if Comuna[i]=="Negrete":
            f249.append(Fecha[i])
            c249.append(Casos[i])
        if Comuna[i]=="Penco":
            f250.append(Fecha[i])
            c250.append(Casos[i])
        if Comuna[i]=="Quilaco":
            f251.append(Fecha[i])
            c251.append(Casos[i])
        if Comuna[i]=="Quilleco":
            f252.append(Fecha[i])
            c252.append(Casos[i])
        if Comuna[i]=="San Pedro de la Paz":
            f253.append(Fecha[i])
            c253.append(Casos[i])
        if Comuna[i]=="San Rosendo":
            f254.append(Fecha[i])
            c254.append(Casos[i])
        if Comuna[i]=="Santa Barbara":
            f255.append(Fecha[i])
            c255.append(Casos[i])
        if Comuna[i]=="Santa Juana":
            f256.append(Fecha[i])
            c256.append(Casos[i])
        if Comuna[i]=="Talcahuano":
            f257.append(Fecha[i])
            c257.append(Casos[i])
        if Comuna[i]=="Tirua":
            f258.append(Fecha[i])
            c258.append(Casos[i])
        if Comuna[i]=="Tome":
            f259.append(Fecha[i])
            c259.append(Casos[i])
        if Comuna[i]=="Tucapel":
            f260.append(Fecha[i])
            c260.append(Casos[i])
        if Comuna[i]=="Yumbel":
            f261.append(Fecha[i])
            c261.append(Casos[i])
        if Comuna[i]=="Desconocido Biobio":
            f262.append(Fecha[i])
            c262.append(Casos[i])
    if Region[i] == "Araucanía":
        if Comuna[i]=="Angol":
            f263.append(Fecha[i])
            c263.append(Casos[i])
        if Comuna[i]=="Carahue":
            f264.append(Fecha[i])
            c264.append(Casos[i])
        if Comuna[i]=="Cholchol":
            f265.append(Fecha[i])
            c265.append(Casos[i])
        if Comuna[i]=="Collipulli":
            f266.append(Fecha[i])
            c266.append(Casos[i])
        if Comuna[i]=="Cunco":
            f267.append(Fecha[i])
            c267.append(Casos[i])
        if Comuna[i]=="Curacautin":
            f268.append(Fecha[i])
            c268.append(Casos[i])
        if Comuna[i]=="Curarrehue":
            f269.append(Fecha[i])
            c269.append(Casos[i])
        if Comuna[i]=="Ercilla":
            f270.append(Fecha[i])
            c270.append(Casos[i])
        if Comuna[i]=="Freire":
            f271.append(Fecha[i])
            c271.append(Casos[i])
        if Comuna[i]=="Galvarino":
            f272.append(Fecha[i])
            c272.append(Casos[i])
        if Comuna[i]=="Gorbea":
            f273.append(Fecha[i])
            c273.append(Casos[i])
        if Comuna[i]=="Lautaro":
            f274.append(Fecha[i])
            c274.append(Casos[i])
        if Comuna[i]=="Loncoche":
            f275.append(Fecha[i])
            c275.append(Casos[i])
        if Comuna[i]=="Lonquimay":
            f276.append(Fecha[i])
            c276.append(Casos[i])
        if Comuna[i]=="Los Sauces":
            f277.append(Fecha[i])
            c277.append(Casos[i])
        if Comuna[i]=="Lumaco":
            f278.append(Fecha[i])
            c278.append(Casos[i])
        if Comuna[i]=="Melipeuco":
            f279.append(Fecha[i])
            c279.append(Casos[i])
        if Comuna[i]=="Nueva Imperial":
            f280.append(Fecha[i])
            c280.append(Casos[i])
        if Comuna[i]=="Padre Las Casas":
            f281.append(Fecha[i])
            c281.append(Casos[i])
        if Comuna[i]=="Perquenco":
            f282.append(Fecha[i])
            c282.append(Casos[i])
        if Comuna[i]=="Pitrufquen":
            f283.append(Fecha[i])
            c283.append(Casos[i])
        if Comuna[i]=="Pucon":
            f284.append(Fecha[i])
            c284.append(Casos[i])
        if Comuna[i]=="Puren":
            f285.append(Fecha[i])
            c285.append(Casos[i])
        if Comuna[i]=="Renaico":
            f286.append(Fecha[i])
            c286.append(Casos[i])
        if Comuna[i]=="Saavedra":
            f287.append(Fecha[i])
            c287.append(Casos[i])
        if Comuna[i]=="Temuco":
            f288.append(Fecha[i])
            c288.append(Casos[i])
        if Comuna[i]=="Teodoro Schmidt":
            f289.append(Fecha[i])
            c289.append(Casos[i])
        if Comuna[i]=="Tolten":
            f290.append(Fecha[i])
            c290.append(Casos[i])
        if Comuna[i]=="Traiguen":
            f291.append(Fecha[i])
            c291.append(Casos[i])
        if Comuna[i]=="Victoria":
            f292.append(Fecha[i])
            c292.append(Casos[i])
        if Comuna[i]=="Vilcun":
            f293.append(Fecha[i])
            c293.append(Casos[i])
        if Comuna[i]=="Villarrica":
            f294.append(Fecha[i])
            c294.append(Casos[i])
        if Comuna[i]=="Desconocido Araucania":
            f295.append(Fecha[i])
            c295.append(Casos[i])
    if Region[i] == "Los Ríos":
        if Comuna[i]=="Corral":
            f296.append(Fecha[i])
            c296.append(Casos[i])
        if Comuna[i]=="Futrono":
            f297.append(Fecha[i])
            c297.append(Casos[i])
        if Comuna[i]=="La Union":
            f298.append(Fecha[i])
            c298.append(Casos[i])
        if Comuna[i]=="Lago Ranco":
            f299.append(Fecha[i])
            c299.append(Casos[i])
        if Comuna[i]=="Lanco":
            f300.append(Fecha[i])
            c300.append(Casos[i])
        if Comuna[i]=="Los Lagos":
            f301.append(Fecha[i])
            c301.append(Casos[i])
        if Comuna[i]=="Mafil":
            f302.append(Fecha[i])
            c302.append(Casos[i])
        if Comuna[i]=="Mariquina":
            f303.append(Fecha[i])
            c303.append(Casos[i])
        if Comuna[i]=="Paillaco":
            f304.append(Fecha[i])
            c304.append(Casos[i])
        if Comuna[i]=="Panguipulli":
            f305.append(Fecha[i])
            c305.append(Casos[i])
        if Comuna[i]=="Rio Bueno":
            f306.append(Fecha[i])
            c306.append(Casos[i])
        if Comuna[i]=="Valdivia":
            f307.append(Fecha[i])
            c307.append(Casos[i])
        if Comuna[i]=="Desconocido Los Rios":
            f308.append(Fecha[i])
            c308.append(Casos[i])
    if Region[i] == "Los Lagos":
        if Comuna[i]=="Ancud":
            f309.append(Fecha[i])
            c309.append(Casos[i])
        if Comuna[i]=="Calbuco":
            f310.append(Fecha[i])
            c310.append(Casos[i])
        if Comuna[i]=="Castro":
            f311.append(Fecha[i])
            c311.append(Casos[i])
        if Comuna[i]=="Chaiten":
            f312.append(Fecha[i])
            c312.append(Casos[i])
        if Comuna[i]=="Chonchi":
            f313.append(Fecha[i])
            c313.append(Casos[i])
        if Comuna[i]=="Cochamo":
            f314.append(Fecha[i])
            c314.append(Casos[i])
        if Comuna[i]=="Curaco de Velez":
            f315.append(Fecha[i])
            c315.append(Casos[i])
        if Comuna[i]=="Fresia":
            f316.append(Fecha[i])
            c316.append(Casos[i])
        if Comuna[i]=="Frutillar":
            f317.append(Fecha[i])
            c317.append(Casos[i])
        if Comuna[i]=="Futaleufu":
            f318.append(Fecha[i])
            c318.append(Casos[i])
        if Comuna[i]=="Hualaihue":
            f319.append(Fecha[i])
            c319.append(Casos[i])
        if Comuna[i]=="Llanquihue":
            f320.append(Fecha[i])
            c320.append(Casos[i])
        if Comuna[i]=="Los Muermos":
            f321.append(Fecha[i])
            c321.append(Casos[i])
        if Comuna[i]=="Maullin":
            f322.append(Fecha[i])
            c322.append(Casos[i])
        if Comuna[i]=="Osorno":
            f323.append(Fecha[i])
            c323.append(Casos[i])
        if Comuna[i]=="Palena":
            f324.append(Fecha[i])
            c324.append(Casos[i])
        if Comuna[i]=="Puerto Montt":
            f325.append(Fecha[i])
            c325.append(Casos[i])
        if Comuna[i]=="Puerto Octay":
            f326.append(Fecha[i])
            c326.append(Casos[i])
        if Comuna[i]=="Puerto Varas":
            f327.append(Fecha[i])
            c327.append(Casos[i])
        if Comuna[i]=="Puqueldon":
            f328.append(Fecha[i])
            c328.append(Casos[i])
        if Comuna[i]=="Purranque":
            f329.append(Fecha[i])
            c329.append(Casos[i])
        if Comuna[i]=="Puyehue":
            f330.append(Fecha[i])
            c330.append(Casos[i])
        if Comuna[i]=="Queilen":
            f331.append(Fecha[i])
            c331.append(Casos[i])
        if Comuna[i]=="Quellon":
            f332.append(Fecha[i])
            c332.append(Casos[i])
        if Comuna[i]=="Quemchi":
            f333.append(Fecha[i])
            c333.append(Casos[i])
        if Comuna[i]=="Quinchao":
            f334.append(Fecha[i])
            c334.append(Casos[i])
        if Comuna[i]=="Rio Negro":
            f335.append(Fecha[i])
            c335.append(Casos[i])
        if Comuna[i]=="San Juan de la Costa":
            f336.append(Fecha[i])
            c336.append(Casos[i])
        if Comuna[i]=="San Pablo":
            f337.append(Fecha[i])
            c337.append(Casos[i])
        if Comuna[i]=="Desconocido Los Lagos":
            f338.append(Fecha[i])
            c338.append(Casos[i])
    if Region[i] == "Aysén":
        if Comuna[i]=="Aysen":
            f339.append(Fecha[i])
            c339.append(Casos[i])
        if Comuna[i]=="Chile Chico":
            f340.append(Fecha[i])
            c340.append(Casos[i])
        if Comuna[i]=="Cisnes":
            f341.append(Fecha[i])
            c341.append(Casos[i])
        if Comuna[i]=="Cochrane":
            f342.append(Fecha[i])
            c342.append(Casos[i])
        if Comuna[i]=="Coyhaique":
            f343.append(Fecha[i])
            c343.append(Casos[i])
        if Comuna[i]=="Guaitecas":
            f344.append(Fecha[i])
            c344.append(Casos[i])
        if Comuna[i]=="Lago Verde":
            f345.append(Fecha[i])
            c345.append(Casos[i])
        if Comuna[i]=="OHiggins":
            f346.append(Fecha[i])
            c346.append(Casos[i])
        if Comuna[i]=="Rio Ibanez":
            f347.append(Fecha[i])
            c347.append(Casos[i])
        if Comuna[i]=="Tortel":
            f348.append(Fecha[i])
            c348.append(Casos[i])
        if Comuna[i]=="Desconocido Aysen":
            f349.append(Fecha[i])
            c349.append(Casos[i])
    if Region[i] == "Magallanes":
        if Comuna[i]=="Antartica":
            f350.append(Fecha[i])
            c350.append(Casos[i])
        if Comuna[i]=="Cabo de Hornos":
            f351.append(Fecha[i])
            c351.append(Casos[i])
        if Comuna[i]=="Laguna Blanca":
            f352.append(Fecha[i])
            c352.append(Casos[i])
        if Comuna[i]=="Natales":
            f353.append(Fecha[i])
            c353.append(Casos[i])
        if Comuna[i]=="Porvenir":
            f354.append(Fecha[i])
            c354.append(Casos[i])
        if Comuna[i]=="Primavera":
            f355.append(Fecha[i])
            c355.append(Casos[i])
        if Comuna[i]=="Punta Arenas":
            f356.append(Fecha[i])
            c356.append(Casos[i])
        if Comuna[i]=="Rio Verde":
            f357.append(Fecha[i])
            c357.append(Casos[i])
        if Comuna[i]=="San Gregorio":
            f358.append(Fecha[i])
            c358.append(Casos[i])
        if Comuna[i]=="Timaukel":
            f359.append(Fecha[i])
            c359.append(Casos[i])
        if Comuna[i]=="Torres del Paine":
            f360.append(Fecha[i])
            c360.append(Casos[i])
        if Comuna[i]=="Desconocido Magallanes":
            f361.append(Fecha[i])
            c361.append(Casos[i])
  

for n in range(1,362):
    exec("z = len(c%s)" % (n))
    for i in range(z):
        if i == 0:
            exec("nc%s.append(c%s[i])" % (n,n))
        else:
            exec("nc%s.append(c%s[i]-c%s[i-1])" %(n,n,n))
for i in range(1,362):
     exec ("df%s = pd.DataFrame()" %(i))
     exec ("df%s['Fecha'] = pd.to_datetime(f%s)" % (i,i))
     exec ("df%s.index = df%s['Fecha']" % (i,i))
     exec ("df%s['Casos'] = nc%s" % (i,i))

print (df288)
n=df288
values = n["Casos"]
print(values)
values = values.astype("int32")

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

mes = df288['2020-09-11':'2020-10-11']
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

for i in range(len(datos_invertidos)):
    datos_invertidos[i] = int(datos_invertidos[i])


prediccion = pd.DataFrame(datos_invertidos)
prediccion.columns = ['pronostico']
prediccion.plot()
plt.show()