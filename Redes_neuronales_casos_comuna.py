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
    mycol = mydb["Comuna_Casos"]
    Region = []
    Fecha =[]
    Casos = []
    Comuna = []

    f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95,f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127,f128,f129,f130,f131,f132,f133,f134,f135,f136,f137,f138,f139,f140,f141,f142,f143,f144,f145,f146,f147,f148,f149,f150,f151,f152,f153,f154,f155,f156,f157,f158,f159,f160,f161,f162,f163,f164,f165,f166,f167,f168,f169,f170,f171,f172,f173,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f184,f185,f186,f187,f188,f189,f190,f191,f192,f193,f194,f195,f196,f197,f198,f199,f200,f201,f202,f203,f204,f205,f206,f207,f208,f209,f210,f211,f212,f213,f214,f215,f216,f217,f218,f219,f220,f221,f222,f223,f224,f225,f226,f227,f228,f229,f230,f231,f232,f233,f234,f235,f236,f237,f238,f239,f240,f241,f242,f243,f244,f245,f246,f247,f248,f249,f250,f251,f252,f253,f254,f255,f256,f257,f258,f259,f260,f261,f262,f263,f264,f265,f266,f267,f268,f269,f270,f271,f272,f273,f274,f275,f276,f277,f278,f279,f280,f281,f282,f283,f284,f285,f286,f287,f288,f289,f290,f291,f292,f293,f294,f295,f296,f297,f298,f299,f300,f301,f302,f303,f304,f305,f306,f307,f308,f309,f310,f311,f312,f313,f314,f315,f316,f317,f318,f319,f320,f321,f322,f323,f324,f325,f326,f327,f328,f329,f330,f331,f332,f333,f334,f335,f336,f337,f338,f339,f340,f341,f342,f343,f344,f345,f346,f347,f348,f349,f350,f351,f352,f353,f354,f355,f356,f357,f358,f359,f360,f361 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74,c75,c76,c77,c78,c79,c80,c81,c82,c83,c84,c85,c86,c87,c88,c89,c90,c91,c92,c93,c94,c95,c96,c97,c98,c99,c100,c101,c102,c103,c104,c105,c106,c107,c108,c109,c110,c111,c112,c113,c114,c115,c116,c117,c118,c119,c120,c121,c122,c123,c124,c125,c126,c127,c128,c129,c130,c131,c132,c133,c134,c135,c136,c137,c138,c139,c140,c141,c142,c143,c144,c145,c146,c147,c148,c149,c150,c151,c152,c153,c154,c155,c156,c157,c158,c159,c160,c161,c162,c163,c164,c165,c166,c167,c168,c169,c170,c171,c172,c173,c174,c175,c176,c177,c178,c179,c180,c181,c182,c183,c184,c185,c186,c187,c188,c189,c190,c191,c192,c193,c194,c195,c196,c197,c198,c199,c200,c201,c202,c203,c204,c205,c206,c207,c208,c209,c210,c211,c212,c213,c214,c215,c216,c217,c218,c219,c220,c221,c222,c223,c224,c225,c226,c227,c228,c229,c230,c231,c232,c233,c234,c235,c236,c237,c238,c239,c240,c241,c242,c243,c244,c245,c246,c247,c248,c249,c250,c251,c252,c253,c254,c255,c256,c257,c258,c259,c260,c261,c262,c263,c264,c265,c266,c267,c268,c269,c270,c271,c272,c273,c274,c275,c276,c277,c278,c279,c280,c281,c282,c283,c284,c285,c286,c287,c288,c289,c290,c291,c292,c293,c294,c295,c296,c297,c298,c299,c300,c301,c302,c303,c304,c305,c306,c307,c308,c309,c310,c311,c312,c313,c314,c315,c316,c317,c318,c319,c320,c321,c322,c323,c324,c325,c326,c327,c328,c329,c330,c331,c332,c333,c334,c335,c336,c337,c338,c339,c340,c341,c342,c343,c344,c345,c346,c347,c348,c349,c350,c351,c352,c353,c354,c355,c356,c357,c358,c359,c360,c361 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    nc1,nc2,nc3,nc4,nc5,nc6,nc7,nc8,nc9,nc10,nc11,nc12,nc13,nc14,nc15,nc16,nc17,nc18,nc19,nc20,nc21,nc22,nc23,nc24,nc25,nc26,nc27,nc28,nc29,nc30,nc31,nc32,nc33,nc34,nc35,nc36,nc37,nc38,nc39,nc40,nc41,nc42,nc43,nc44,nc45,nc46,nc47,nc48,nc49,nc50,nc51,nc52,nc53,nc54,nc55,nc56,nc57,nc58,nc59,nc60,nc61,nc62,nc63,nc64,nc65,nc66,nc67,nc68,nc69,nc70,nc71,nc72,nc73,nc74,nc75,nc76,nc77,nc78,nc79,nc80,nc81,nc82,nc83,nc84,nc85,nc86,nc87,nc88,nc89,nc90,nc91,nc92,nc93,nc94,nc95,nc96,nc97,nc98,nc99,nc100,nc101,nc102,nc103,nc104,nc105,nc106,nc107,nc108,nc109,nc110,nc111,nc112,nc113,nc114,nc115,nc116,nc117,nc118,nc119,nc120,nc121,nc122,nc123,nc124,nc125,nc126,nc127,nc128,nc129,nc130,nc131,nc132,nc133,nc134,nc135,nc136,nc137,nc138,nc139,nc140,nc141,nc142,nc143,nc144,nc145,nc146,nc147,nc148,nc149,nc150,nc151,nc152,nc153,nc154,nc155,nc156,nc157,nc158,nc159,nc160,nc161,nc162,nc163,nc164,nc165,nc166,nc167,nc168,nc169,nc170,nc171,nc172,nc173,nc174,nc175,nc176,nc177,nc178,nc179,nc180,nc181,nc182,nc183,nc184,nc185,nc186,nc187,nc188,nc189,nc190,nc191,nc192,nc193,nc194,nc195,nc196,nc197,nc198,nc199,nc200,nc201,nc202,nc203,nc204,nc205,nc206,nc207,nc208,nc209,nc210,nc211,nc212,nc213,nc214,nc215,nc216,nc217,nc218,nc219,nc220,nc221,nc222,nc223,nc224,nc225,nc226,nc227,nc228,nc229,nc230,nc231,nc232,nc233,nc234,nc235,nc236,nc237,nc238,nc239,nc240,nc241,nc242,nc243,nc244,nc245,nc246,nc247,nc248,nc249,nc250,nc251,nc252,nc253,nc254,nc255,nc256,nc257,nc258,nc259,nc260,nc261,nc262,nc263,nc264,nc265,nc266,nc267,nc268,nc269,nc270,nc271,nc272,nc273,nc274,nc275,nc276,nc277,nc278,nc279,nc280,nc281,nc282,nc283,nc284,nc285,nc286,nc287,nc288,nc289,nc290,nc291,nc292,nc293,nc294,nc295,nc296,nc297,nc298,nc299,nc300,nc301,nc302,nc303,nc304,nc305,nc306,nc307,nc308,nc309,nc310,nc311,nc312,nc313,nc314,nc315,nc316,nc317,nc318,nc319,nc320,nc321,nc322,nc323,nc324,nc325,nc326,nc327,nc328,nc329,nc330,nc331,nc332,nc333,nc334,nc335,nc336,nc337,nc338,nc339,nc340,nc341,nc342,nc343,nc344,nc345,nc346,nc347,nc348,nc349,nc350,nc351,nc352,nc353,nc354,nc355,nc356,nc357,nc358,nc359,nc360,nc361 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

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
    
        """
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
        """
    
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
    if opcion == 8:
        for i in range(len(c8)):
            if i == 0:
                nc8.append(c8[i])
            else:
                nc8.append(c8[i]-c8[i-1])
        df8 = pd.DataFrame()
        df8['Fecha'] = pd.to_datetime(f8)
        df8.index = df8['Fecha']
        df8['Casos'] = nc8
        n=df8
    if opcion == 9:
        for i in range(len(c9)):
            if i == 0:
                nc9.append(c9[i])
            else:
                nc9.append(c9[i]-c9[i-1])
        df9 = pd.DataFrame()
        df9['Fecha'] = pd.to_datetime(f9)
        df9.index = df9['Fecha']
        df9['Casos'] = nc9
        n=df9
    if opcion == 10:
        for i in range(len(c10)):
            if i == 0:
                nc10.append(c10[i])
            else:
                nc10.append(c10[i]-c10[i-1])
        df10 = pd.DataFrame()
        df10['Fecha'] = pd.to_datetime(f10)
        df10.index = df10['Fecha']
        df10['Casos'] = nc10
        n=df10
    if opcion == 11:
        for i in range(len(c11)):
            if i == 0:
                nc11.append(c11[i])
            else:
                nc11.append(c11[i]-c11[i-1])
        df11 = pd.DataFrame()
        df11['Fecha'] = pd.to_datetime(f11)
        df11.index = df11['Fecha']
        df11['Casos'] = nc11
        n=df11
    if opcion == 12:
        for i in range(len(c12)):
            if i == 0:
                nc12.append(c12[i])
            else:
                nc12.append(c12[i]-c12[i-1])
        df12 = pd.DataFrame()
        df12['Fecha'] = pd.to_datetime(f12)
        df12.index = df12['Fecha']
        df12['Casos'] = nc12
        n=df12
    if opcion == 13:
        for i in range(len(c13)):
            if i == 0:
                nc13.append(c13[i])
            else:
                nc13.append(c13[i]-c13[i-1])
        df13 = pd.DataFrame()
        df13['Fecha'] = pd.to_datetime(f13)
        df13.index = df13['Fecha']
        df13['Casos'] = nc13
        n=df13
    if opcion == 14:
        for i in range(len(c14)):
            if i == 0:
                nc14.append(c14[i])
            else:
                nc14.append(c14[i]-c14[i-1])
        df14 = pd.DataFrame()
        df14['Fecha'] = pd.to_datetime(f14)
        df14.index = df14['Fecha']
        df14['Casos'] = nc14
        n=df14
    if opcion == 15:
        for i in range(len(c15)):
            if i == 0:
                nc15.append(c15[i])
            else:
                nc15.append(c15[i]-c15[i-1])
        df15 = pd.DataFrame()
        df15['Fecha'] = pd.to_datetime(f15)
        df15.index = df15['Fecha']
        df15['Casos'] = nc15
        n=df15
    if opcion == 16:
        for i in range(len(c16)):
            if i == 0:
                nc16.append(c16[i])
            else:
                nc16.append(c16[i]-c16[i-1])
        df16 = pd.DataFrame()
        df16['Fecha'] = pd.to_datetime(f16)
        df16.index = df16['Fecha']
        df16['Casos'] = nc16
        n=df16
    if opcion == 17:
        for i in range(len(c17)):
            if i == 0:
                nc17.append(c17[i])
            else:
                nc17.append(c17[i]-c17[i-1])
        df17 = pd.DataFrame()
        df17['Fecha'] = pd.to_datetime(f17)
        df17.index = df17['Fecha']
        df17['Casos'] = nc17
        n=df17
    if opcion == 18:
        for i in range(len(c18)):
            if i == 0:
                nc18.append(c18[i])
            else:
                nc18.append(c18[i]-c18[i-1])
        df18 = pd.DataFrame()
        df18['Fecha'] = pd.to_datetime(f18)
        df18.index = df18['Fecha']
        df18['Casos'] = nc18
        n=df18
    if opcion == 19:
        for i in range(len(c19)):
            if i == 0:
                nc19.append(c19[i])
            else:
                nc19.append(c19[i]-c19[i-1])
        df19 = pd.DataFrame()
        df19['Fecha'] = pd.to_datetime(f19)
        df19.index = df19['Fecha']
        df19['Casos'] = nc19
        n=df19
    if opcion == 20:
        for i in range(len(c20)):
            if i == 0:
                nc20.append(c20[i])
            else:
                nc20.append(c20[i]-c20[i-1])
        df20 = pd.DataFrame()
        df20['Fecha'] = pd.to_datetime(f20)
        df20.index = df20['Fecha']
        df20['Casos'] = nc20
        n=df20
    if opcion == 21:
        for i in range(len(c21)):
            if i == 0:
                nc21.append(c21[i])
            else:
                nc21.append(c21[i]-c21[i-1])
        df21 = pd.DataFrame()
        df21['Fecha'] = pd.to_datetime(f21)
        df21.index = df21['Fecha']
        df21['Casos'] = nc21
        n=df21
    if opcion == 22:
        for i in range(len(c22)):
            if i == 0:
                nc22.append(c22[i])
            else:
                nc22.append(c22[i]-c22[i-1])
        df22 = pd.DataFrame()
        df22['Fecha'] = pd.to_datetime(f22)
        df22.index = df22['Fecha']
        df22['Casos'] = nc22
        n=df22
    if opcion == 23:
        for i in range(len(c23)):
            if i == 0:
                nc23.append(c23[i])
            else:
                nc23.append(c23[i]-c23[i-1])
        df23 = pd.DataFrame()
        df23['Fecha'] = pd.to_datetime(f23)
        df23.index = df23['Fecha']
        df23['Casos'] = nc23
        n=df23
    if opcion == 24:
        for i in range(len(c24)):
            if i == 0:
                nc24.append(c24[i])
            else:
                nc24.append(c24[i]-c24[i-1])
        df24 = pd.DataFrame()
        df24['Fecha'] = pd.to_datetime(f24)
        df24.index = df24['Fecha']
        df24['Casos'] = nc24
        n=df24
    if opcion == 25:
        for i in range(len(c25)):
            if i == 0:
                nc25.append(c25[i])
            else:
                nc25.append(c25[i]-c25[i-1])
        df25 = pd.DataFrame()
        df25['Fecha'] = pd.to_datetime(f25)
        df25.index = df25['Fecha']
        df25['Casos'] = nc25
        n=df25
    if opcion == 26:
        for i in range(len(c26)):
            if i == 0:
                nc26.append(c26[i])
            else:
                nc26.append(c26[i]-c26[i-1])
        df26 = pd.DataFrame()
        df26['Fecha'] = pd.to_datetime(f26)
        df26.index = df26['Fecha']
        df26['Casos'] = nc26
        n=df26
    if opcion == 27:
        for i in range(len(c27)):
            if i == 0:
                nc27.append(c27[i])
            else:
                nc27.append(c27[i]-c27[i-1])
        df27 = pd.DataFrame()
        df27['Fecha'] = pd.to_datetime(f27)
        df27.index = df27['Fecha']
        df27['Casos'] = nc27
        n=df27
    if opcion == 28:
        for i in range(len(c28)):
            if i == 0:
                nc28.append(c28[i])
            else:
                nc28.append(c28[i]-c28[i-1])
        df28 = pd.DataFrame()
        df28['Fecha'] = pd.to_datetime(f28)
        df28.index = df28['Fecha']
        df28['Casos'] = nc28
        n=df28
    if opcion == 29:
        for i in range(len(c29)):
            if i == 0:
                nc29.append(c29[i])
            else:
                nc29.append(c29[i]-c29[i-1])
        df29 = pd.DataFrame()
        df29['Fecha'] = pd.to_datetime(f29)
        df29.index = df29['Fecha']
        df29['Casos'] = nc29
        n=df29
    if opcion == 30:
        for i in range(len(c30)):
            if i == 0:
                nc30.append(c30[i])
            else:
                nc30.append(c30[i]-c30[i-1])
        df30 = pd.DataFrame()
        df30['Fecha'] = pd.to_datetime(f30)
        df30.index = df30['Fecha']
        df30['Casos'] = nc30
        n=df30
    if opcion == 31:
        for i in range(len(c31)):
            if i == 0:
                nc31.append(c31[i])
            else:
                nc31.append(c31[i]-c31[i-1])
        df31 = pd.DataFrame()
        df31['Fecha'] = pd.to_datetime(f31)
        df31.index = df31['Fecha']
        df31['Casos'] = nc31
        n=df31
    if opcion == 32:
        for i in range(len(c32)):
            if i == 0:
                nc32.append(c32[i])
            else:
                nc32.append(c32[i]-c32[i-1])
        df32 = pd.DataFrame()
        df32['Fecha'] = pd.to_datetime(f32)
        df32.index = df32['Fecha']
        df32['Casos'] = nc32
        n=df32
    if opcion == 33:
        for i in range(len(c33)):
            if i == 0:
                nc33.append(c33[i])
            else:
                nc33.append(c33[i]-c33[i-1])
        df33 = pd.DataFrame()
        df33['Fecha'] = pd.to_datetime(f33)
        df33.index = df33['Fecha']
        df33['Casos'] = nc33
        n=df33
    if opcion == 34:
        for i in range(len(c34)):
            if i == 0:
                nc34.append(c34[i])
            else:
                nc34.append(c34[i]-c34[i-1])
        df34 = pd.DataFrame()
        df34['Fecha'] = pd.to_datetime(f34)
        df34.index = df34['Fecha']
        df34['Casos'] = nc34
        n=df34
    if opcion == 35:
        for i in range(len(c35)):
            if i == 0:
                nc35.append(c35[i])
            else:
                nc35.append(c35[i]-c35[i-1])
        df35 = pd.DataFrame()
        df35['Fecha'] = pd.to_datetime(f35)
        df35.index = df35['Fecha']
        df35['Casos'] = nc35
        n=df35
    if opcion == 36:
        for i in range(len(c36)):
            if i == 0:
                nc36.append(c36[i])
            else:
                nc36.append(c36[i]-c36[i-1])
        df36 = pd.DataFrame()
        df36['Fecha'] = pd.to_datetime(f36)
        df36.index = df36['Fecha']
        df36['Casos'] = nc36
        n=df36
    if opcion == 37:
        for i in range(len(c37)):
            if i == 0:
                nc37.append(c37[i])
            else:
                nc37.append(c37[i]-c37[i-1])
        df37 = pd.DataFrame()
        df37['Fecha'] = pd.to_datetime(f37)
        df37.index = df37['Fecha']
        df37['Casos'] = nc37
        n=df37
    if opcion == 38:
        for i in range(len(c38)):
            if i == 0:
                nc38.append(c38[i])
            else:
                nc38.append(c38[i]-c38[i-1])
        df38 = pd.DataFrame()
        df38['Fecha'] = pd.to_datetime(f38)
        df38.index = df38['Fecha']
        df38['Casos'] = nc38
        n=df38
    if opcion == 39:
        for i in range(len(c39)):
            if i == 0:
                nc39.append(c39[i])
            else:
                nc39.append(c39[i]-c39[i-1])
        df39 = pd.DataFrame()
        df39['Fecha'] = pd.to_datetime(f39)
        df39.index = df39['Fecha']
        df39['Casos'] = nc39
        n=df39
    if opcion == 40:
        for i in range(len(c40)):
            if i == 0:
                nc40.append(c40[i])
            else:
                nc40.append(c40[i]-c40[i-1])
        df40 = pd.DataFrame()
        df40['Fecha'] = pd.to_datetime(f40)
        df40.index = df40['Fecha']
        df40['Casos'] = nc40
        n=df40
    if opcion == 41:
        for i in range(len(c41)):
            if i == 0:
                nc41.append(c41[i])
            else:
                nc41.append(c41[i]-c41[i-1])
        df41 = pd.DataFrame()
        df41['Fecha'] = pd.to_datetime(f41)
        df41.index = df41['Fecha']
        df41['Casos'] = nc41
        n=df41
    if opcion == 42:
        for i in range(len(c42)):
            if i == 0:
                nc42.append(c42[i])
            else:
                nc42.append(c42[i]-c42[i-1])
        df42 = pd.DataFrame()
        df42['Fecha'] = pd.to_datetime(f42)
        df42.index = df42['Fecha']
        df42['Casos'] = nc42
        n=df42
    if opcion == 43:
        for i in range(len(c43)):
            if i == 0:
                nc43.append(c43[i])
            else:
                nc43.append(c43[i]-c43[i-1])
        df43 = pd.DataFrame()
        df43['Fecha'] = pd.to_datetime(f43)
        df43.index = df43['Fecha']
        df43['Casos'] = nc43
        n=df43
    if opcion == 44:
        for i in range(len(c44)):
            if i == 0:
                nc44.append(c44[i])
            else:
                nc44.append(c44[i]-c44[i-1])
        df44 = pd.DataFrame()
        df44['Fecha'] = pd.to_datetime(f44)
        df44.index = df44['Fecha']
        df44['Casos'] = nc44
        n=df44
    if opcion == 45:
        for i in range(len(c45)):
            if i == 0:
                nc45.append(c45[i])
            else:
                nc45.append(c45[i]-c45[i-1])
        df45 = pd.DataFrame()
        df45['Fecha'] = pd.to_datetime(f45)
        df45.index = df45['Fecha']
        df45['Casos'] = nc45
        n=df45
    if opcion == 46:
        for i in range(len(c46)):
            if i == 0:
                nc46.append(c46[i])
            else:
                nc46.append(c46[i]-c46[i-1])
        df46 = pd.DataFrame()
        df46['Fecha'] = pd.to_datetime(f46)
        df46.index = df46['Fecha']
        df46['Casos'] = nc46
        n=df46
    if opcion == 47:
        for i in range(len(c47)):
            if i == 0:
                nc47.append(c47[i])
            else:
                nc47.append(c47[i]-c47[i-1])
        df47 = pd.DataFrame()
        df47['Fecha'] = pd.to_datetime(f47)
        df47.index = df47['Fecha']
        df47['Casos'] = nc47
        n=df47
    if opcion == 48:
        for i in range(len(c48)):
            if i == 0:
                nc48.append(c48[i])
            else:
                nc48.append(c48[i]-c48[i-1])
        df48 = pd.DataFrame()
        df48['Fecha'] = pd.to_datetime(f48)
        df48.index = df48['Fecha']
        df48['Casos'] = nc48
        n=df48
    if opcion == 49:
        for i in range(len(c49)):
            if i == 0:
                nc49.append(c49[i])
            else:
                nc49.append(c49[i]-c49[i-1])
        df49 = pd.DataFrame()
        df49['Fecha'] = pd.to_datetime(f49)
        df49.index = df49['Fecha']
        df49['Casos'] = nc49
        n=df49
    if opcion == 50:
        for i in range(len(c50)):
            if i == 0:
                nc50.append(c50[i])
            else:
                nc50.append(c50[i]-c50[i-1])
        df50 = pd.DataFrame()
        df50['Fecha'] = pd.to_datetime(f50)
        df50.index = df50['Fecha']
        df50['Casos'] = nc50
        n=df50
    if opcion == 51:
        for i in range(len(c51)):
            if i == 0:
                nc51.append(c51[i])
            else:
                nc51.append(c51[i]-c51[i-1])
        df51 = pd.DataFrame()
        df51['Fecha'] = pd.to_datetime(f51)
        df51.index = df51['Fecha']
        df51['Casos'] = nc51
        n=df51
    if opcion == 52:
        for i in range(len(c52)):
            if i == 0:
                nc52.append(c52[i])
            else:
                nc52.append(c52[i]-c52[i-1])
        df52 = pd.DataFrame()
        df52['Fecha'] = pd.to_datetime(f52)
        df52.index = df52['Fecha']
        df52['Casos'] = nc52
        n=df52
    if opcion == 53:
        for i in range(len(c53)):
            if i == 0:
                nc53.append(c53[i])
            else:
                nc53.append(c53[i]-c53[i-1])
        df53 = pd.DataFrame()
        df53['Fecha'] = pd.to_datetime(f53)
        df53.index = df53['Fecha']
        df53['Casos'] = nc53
        n=df53
    if opcion == 54:
        for i in range(len(c54)):
            if i == 0:
                nc54.append(c54[i])
            else:
                nc54.append(c54[i]-c54[i-1])
        df54 = pd.DataFrame()
        df54['Fecha'] = pd.to_datetime(f54)
        df54.index = df54['Fecha']
        df54['Casos'] = nc54
        n=df54
    if opcion == 55:
        for i in range(len(c55)):
            if i == 0:
                nc55.append(c55[i])
            else:
                nc55.append(c55[i]-c55[i-1])
        df55 = pd.DataFrame()
        df55['Fecha'] = pd.to_datetime(f55)
        df55.index = df55['Fecha']
        df55['Casos'] = nc55
        n=df55
    if opcion == 56:
        for i in range(len(c56)):
            if i == 0:
                nc56.append(c56[i])
            else:
                nc56.append(c56[i]-c56[i-1])
        df56 = pd.DataFrame()
        df56['Fecha'] = pd.to_datetime(f56)
        df56.index = df56['Fecha']
        df56['Casos'] = nc56
        n=df56
    if opcion == 57:
        for i in range(len(c57)):
            if i == 0:
                nc57.append(c57[i])
            else:
                nc57.append(c57[i]-c57[i-1])
        df57 = pd.DataFrame()
        df57['Fecha'] = pd.to_datetime(f57)
        df57.index = df57['Fecha']
        df57['Casos'] = nc57
        n=df57
    if opcion == 58:
        for i in range(len(c58)):
            if i == 0:
                nc58.append(c58[i])
            else:
                nc58.append(c58[i]-c58[i-1])
        df58 = pd.DataFrame()
        df58['Fecha'] = pd.to_datetime(f58)
        df58.index = df58['Fecha']
        df58['Casos'] = nc58
        n=df58
    if opcion == 59:
        for i in range(len(c59)):
            if i == 0:
                nc59.append(c59[i])
            else:
                nc59.append(c59[i]-c59[i-1])
        df59 = pd.DataFrame()
        df59['Fecha'] = pd.to_datetime(f59)
        df59.index = df59['Fecha']
        df59['Casos'] = nc59
        n=df59
    if opcion == 60:
        for i in range(len(c60)):
            if i == 0:
                nc60.append(c60[i])
            else:
                nc60.append(c60[i]-c60[i-1])
        df60 = pd.DataFrame()
        df60['Fecha'] = pd.to_datetime(f60)
        df60.index = df60['Fecha']
        df60['Casos'] = nc60
        n=df60
    if opcion == 61:
        for i in range(len(c61)):
            if i == 0:
                nc61.append(c61[i])
            else:
                nc61.append(c61[i]-c61[i-1])
        df61 = pd.DataFrame()
        df61['Fecha'] = pd.to_datetime(f61)
        df61.index = df61['Fecha']
        df61['Casos'] = nc61
        n=df61
    if opcion == 62:
        for i in range(len(c62)):
            if i == 0:
                nc62.append(c62[i])
            else:
                nc62.append(c62[i]-c62[i-1])
        df62 = pd.DataFrame()
        df62['Fecha'] = pd.to_datetime(f62)
        df62.index = df62['Fecha']
        df62['Casos'] = nc62
        n=df62
    if opcion == 63:
        for i in range(len(c63)):
            if i == 0:
                nc63.append(c63[i])
            else:
                nc63.append(c63[i]-c63[i-1])
        df63 = pd.DataFrame()
        df63['Fecha'] = pd.to_datetime(f63)
        df63.index = df63['Fecha']
        df63['Casos'] = nc63
        n=df63
    if opcion == 64:
        for i in range(len(c64)):
            if i == 0:
                nc64.append(c64[i])
            else:
                nc64.append(c64[i]-c64[i-1])
        df64 = pd.DataFrame()
        df64['Fecha'] = pd.to_datetime(f64)
        df64.index = df64['Fecha']
        df64['Casos'] = nc64
        n=df64
    if opcion == 65:
        for i in range(len(c65)):
            if i == 0:
                nc65.append(c65[i])
            else:
                nc65.append(c65[i]-c65[i-1])
        df65 = pd.DataFrame()
        df65['Fecha'] = pd.to_datetime(f65)
        df65.index = df65['Fecha']
        df65['Casos'] = nc65
        n=df65
    if opcion == 66:
        for i in range(len(c66)):
            if i == 0:
                nc66.append(c66[i])
            else:
                nc66.append(c66[i]-c66[i-1])
        df66 = pd.DataFrame()
        df66['Fecha'] = pd.to_datetime(f66)
        df66.index = df66['Fecha']
        df66['Casos'] = nc66
        n=df66
    if opcion == 67:
        for i in range(len(c67)):
            if i == 0:
                nc67.append(c67[i])
            else:
                nc67.append(c67[i]-c67[i-1])
        df67 = pd.DataFrame()
        df67['Fecha'] = pd.to_datetime(f67)
        df67.index = df67['Fecha']
        df67['Casos'] = nc67
        n=df67
    if opcion == 68:
        for i in range(len(c68)):
            if i == 0:
                nc68.append(c68[i])
            else:
                nc68.append(c68[i]-c68[i-1])
        df68 = pd.DataFrame()
        df68['Fecha'] = pd.to_datetime(f68)
        df68.index = df68['Fecha']
        df68['Casos'] = nc68
        n=df68
    if opcion == 69:
        for i in range(len(c69)):
            if i == 0:
                nc69.append(c69[i])
            else:
                nc69.append(c69[i]-c69[i-1])
        df69 = pd.DataFrame()
        df69['Fecha'] = pd.to_datetime(f69)
        df69.index = df69['Fecha']
        df69['Casos'] = nc69
        n=df69
    if opcion == 70:
        for i in range(len(c70)):
            if i == 0:
                nc70.append(c70[i])
            else:
                nc70.append(c70[i]-c70[i-1])
        df70 = pd.DataFrame()
        df70['Fecha'] = pd.to_datetime(f70)
        df70.index = df70['Fecha']
        df70['Casos'] = nc70
        n=df70
    if opcion == 71:
        for i in range(len(c71)):
            if i == 0:
                nc71.append(c71[i])
            else:
                nc71.append(c71[i]-c71[i-1])
        df71 = pd.DataFrame()
        df71['Fecha'] = pd.to_datetime(f71)
        df71.index = df71['Fecha']
        df71['Casos'] = nc71
        n=df71
    if opcion == 72:
        for i in range(len(c72)):
            if i == 0:
                nc72.append(c72[i])
            else:
                nc72.append(c72[i]-c72[i-1])
        df72 = pd.DataFrame()
        df72['Fecha'] = pd.to_datetime(f72)
        df72.index = df72['Fecha']
        df72['Casos'] = nc72
        n=df72
    if opcion == 73:
        for i in range(len(c73)):
            if i == 0:
                nc73.append(c73[i])
            else:
                nc73.append(c73[i]-c73[i-1])
        df73 = pd.DataFrame()
        df73['Fecha'] = pd.to_datetime(f73)
        df73.index = df73['Fecha']
        df73['Casos'] = nc73
        n=df73
    if opcion == 74:
        for i in range(len(c74)):
            if i == 0:
                nc74.append(c74[i])
            else:
                nc74.append(c74[i]-c74[i-1])
        df74 = pd.DataFrame()
        df74['Fecha'] = pd.to_datetime(f74)
        df74.index = df74['Fecha']
        df74['Casos'] = nc74
        n=df74
    if opcion == 75:
        for i in range(len(c75)):
            if i == 0:
                nc75.append(c75[i])
            else:
                nc75.append(c75[i]-c75[i-1])
        df75 = pd.DataFrame()
        df75['Fecha'] = pd.to_datetime(f75)
        df75.index = df75['Fecha']
        df75['Casos'] = nc75
        n=df75
    if opcion == 76:
        for i in range(len(c76)):
            if i == 0:
                nc76.append(c76[i])
            else:
                nc76.append(c76[i]-c76[i-1])
        df76 = pd.DataFrame()
        df76['Fecha'] = pd.to_datetime(f76)
        df76.index = df76['Fecha']
        df76['Casos'] = nc76
        n=df76
    if opcion == 77:
        for i in range(len(c77)):
            if i == 0:
                nc77.append(c77[i])
            else:
                nc77.append(c77[i]-c77[i-1])
        df77 = pd.DataFrame()
        df77['Fecha'] = pd.to_datetime(f77)
        df77.index = df77['Fecha']
        df77['Casos'] = nc77
        n=df77
    if opcion == 78:
        for i in range(len(c78)):
            if i == 0:
                nc78.append(c78[i])
            else:
                nc78.append(c78[i]-c78[i-1])
        df78 = pd.DataFrame()
        df78['Fecha'] = pd.to_datetime(f78)
        df78.index = df78['Fecha']
        df78['Casos'] = nc78
        n=df78
    if opcion == 79:
        for i in range(len(c79)):
            if i == 0:
                nc79.append(c79[i])
            else:
                nc79.append(c79[i]-c79[i-1])
        df79 = pd.DataFrame()
        df79['Fecha'] = pd.to_datetime(f79)
        df79.index = df79['Fecha']
        df79['Casos'] = nc79
        n=df79
    if opcion == 80:
        for i in range(len(c80)):
            if i == 0:
                nc80.append(c80[i])
            else:
                nc80.append(c80[i]-c80[i-1])
        df80 = pd.DataFrame()
        df80['Fecha'] = pd.to_datetime(f80)
        df80.index = df80['Fecha']
        df80['Casos'] = nc80
        n=df80
    if opcion == 81:
        for i in range(len(c81)):
            if i == 0:
                nc81.append(c81[i])
            else:
                nc81.append(c81[i]-c81[i-1])
        df81 = pd.DataFrame()
        df81['Fecha'] = pd.to_datetime(f81)
        df81.index = df81['Fecha']
        df81['Casos'] = nc81
        n=df81
    if opcion == 82:
        for i in range(len(c82)):
            if i == 0:
                nc82.append(c82[i])
            else:
                nc82.append(c82[i]-c82[i-1])
        df82 = pd.DataFrame()
        df82['Fecha'] = pd.to_datetime(f82)
        df82.index = df82['Fecha']
        df82['Casos'] = nc82
        n=df82
    if opcion == 83:
        for i in range(len(c83)):
            if i == 0:
                nc83.append(c83[i])
            else:
                nc83.append(c83[i]-c83[i-1])
        df83 = pd.DataFrame()
        df83['Fecha'] = pd.to_datetime(f83)
        df83.index = df83['Fecha']
        df83['Casos'] = nc83
        n=df83
    if opcion == 84:
        for i in range(len(c84)):
            if i == 0:
                nc84.append(c84[i])
            else:
                nc84.append(c84[i]-c84[i-1])
        df84 = pd.DataFrame()
        df84['Fecha'] = pd.to_datetime(f84)
        df84.index = df84['Fecha']
        df84['Casos'] = nc84
        n=df84
    if opcion == 85:
        for i in range(len(c85)):
            if i == 0:
                nc85.append(c85[i])
            else:
                nc85.append(c85[i]-c85[i-1])
        df85 = pd.DataFrame()
        df85['Fecha'] = pd.to_datetime(f85)
        df85.index = df85['Fecha']
        df85['Casos'] = nc85
        n=df85
    if opcion == 86:
        for i in range(len(c86)):
            if i == 0:
                nc86.append(c86[i])
            else:
                nc86.append(c86[i]-c86[i-1])
        df86 = pd.DataFrame()
        df86['Fecha'] = pd.to_datetime(f86)
        df86.index = df86['Fecha']
        df86['Casos'] = nc86
        n=df86
    if opcion == 87:
        for i in range(len(c87)):
            if i == 0:
                nc87.append(c87[i])
            else:
                nc87.append(c87[i]-c87[i-1])
        df87 = pd.DataFrame()
        df87['Fecha'] = pd.to_datetime(f87)
        df87.index = df87['Fecha']
        df87['Casos'] = nc87
        n=df87
    if opcion == 88:
        for i in range(len(c88)):
            if i == 0:
                nc88.append(c88[i])
            else:
                nc88.append(c88[i]-c88[i-1])
        df88 = pd.DataFrame()
        df88['Fecha'] = pd.to_datetime(f88)
        df88.index = df88['Fecha']
        df88['Casos'] = nc88
        n=df88
    if opcion == 89:
        for i in range(len(c89)):
            if i == 0:
                nc89.append(c89[i])
            else:
                nc89.append(c89[i]-c89[i-1])
        df89 = pd.DataFrame()
        df89['Fecha'] = pd.to_datetime(f89)
        df89.index = df89['Fecha']
        df89['Casos'] = nc89
        n=df89
    if opcion == 90:
        for i in range(len(c90)):
            if i == 0:
                nc90.append(c90[i])
            else:
                nc90.append(c90[i]-c90[i-1])
        df90 = pd.DataFrame()
        df90['Fecha'] = pd.to_datetime(f90)
        df90.index = df90['Fecha']
        df90['Casos'] = nc90
        n=df90
    if opcion == 91:
        for i in range(len(c91)):
            if i == 0:
                nc91.append(c91[i])
            else:
                nc91.append(c91[i]-c91[i-1])
        df91 = pd.DataFrame()
        df91['Fecha'] = pd.to_datetime(f91)
        df91.index = df91['Fecha']
        df91['Casos'] = nc91
        n=df91
    if opcion == 92:
        for i in range(len(c92)):
            if i == 0:
                nc92.append(c92[i])
            else:
                nc92.append(c92[i]-c92[i-1])
        df92 = pd.DataFrame()
        df92['Fecha'] = pd.to_datetime(f92)
        df92.index = df92['Fecha']
        df92['Casos'] = nc92
        n=df92
    if opcion == 93:
        for i in range(len(c93)):
            if i == 0:
                nc93.append(c93[i])
            else:
                nc93.append(c93[i]-c93[i-1])
        df93 = pd.DataFrame()
        df93['Fecha'] = pd.to_datetime(f93)
        df93.index = df93['Fecha']
        df93['Casos'] = nc93
        n=df93
    if opcion == 94:
        for i in range(len(c94)):
            if i == 0:
                nc94.append(c94[i])
            else:
                nc94.append(c94[i]-c94[i-1])
        df94 = pd.DataFrame()
        df94['Fecha'] = pd.to_datetime(f94)
        df94.index = df94['Fecha']
        df94['Casos'] = nc94
        n=df94
    if opcion == 95:
        for i in range(len(c95)):
            if i == 0:
                nc95.append(c95[i])
            else:
                nc95.append(c95[i]-c95[i-1])
        df95 = pd.DataFrame()
        df95['Fecha'] = pd.to_datetime(f95)
        df95.index = df95['Fecha']
        df95['Casos'] = nc95
        n=df95
    if opcion == 96:
        for i in range(len(c96)):
            if i == 0:
                nc96.append(c96[i])
            else:
                nc96.append(c96[i]-c96[i-1])
        df96 = pd.DataFrame()
        df96['Fecha'] = pd.to_datetime(f96)
        df96.index = df96['Fecha']
        df96['Casos'] = nc96
        n=df96
    if opcion == 97:
        for i in range(len(c97)):
            if i == 0:
                nc97.append(c97[i])
            else:
                nc97.append(c97[i]-c97[i-1])
        df97 = pd.DataFrame()
        df97['Fecha'] = pd.to_datetime(f97)
        df97.index = df97['Fecha']
        df97['Casos'] = nc97
        n=df97
    if opcion == 98:
        for i in range(len(c98)):
            if i == 0:
                nc98.append(c98[i])
            else:
                nc98.append(c98[i]-c98[i-1])
        df98 = pd.DataFrame()
        df98['Fecha'] = pd.to_datetime(f98)
        df98.index = df98['Fecha']
        df98['Casos'] = nc98
        n=df98
    if opcion == 99:
        for i in range(len(c99)):
            if i == 0:
                nc99.append(c99[i])
            else:
                nc99.append(c99[i]-c99[i-1])
        df99 = pd.DataFrame()
        df99['Fecha'] = pd.to_datetime(f99)
        df99.index = df99['Fecha']
        df99['Casos'] = nc99
        n=df99
    if opcion == 100:
        for i in range(len(c100)):
            if i == 0:
                nc100.append(c100[i])
            else:
                nc100.append(c100[i]-c100[i-1])
        df100 = pd.DataFrame()
        df100['Fecha'] = pd.to_datetime(f100)
        df100.index = df100['Fecha']
        df100['Casos'] = nc100
        n=df100
    if opcion == 101:
        for i in range(len(c101)):
            if i == 0:
                nc101.append(c101[i])
            else:
                nc101.append(c101[i]-c101[i-1])
        df101 = pd.DataFrame()
        df101['Fecha'] = pd.to_datetime(f101)
        df101.index = df101['Fecha']
        df101['Casos'] = nc101
        n=df101
    if opcion == 102:
        for i in range(len(c102)):
            if i == 0:
                nc102.append(c102[i])
            else:
                nc102.append(c102[i]-c102[i-1])
        df102 = pd.DataFrame()
        df102['Fecha'] = pd.to_datetime(f102)
        df102.index = df102['Fecha']
        df102['Casos'] = nc102
        n=df102
    if opcion == 103:
        for i in range(len(c103)):
            if i == 0:
                nc103.append(c103[i])
            else:
                nc103.append(c103[i]-c103[i-1])
        df103 = pd.DataFrame()
        df103['Fecha'] = pd.to_datetime(f103)
        df103.index = df103['Fecha']
        df103['Casos'] = nc103
        n=df103
    if opcion == 104:
        for i in range(len(c104)):
            if i == 0:
                nc104.append(c104[i])
            else:
                nc104.append(c104[i]-c104[i-1])
        df104 = pd.DataFrame()
        df104['Fecha'] = pd.to_datetime(f104)
        df104.index = df104['Fecha']
        df104['Casos'] = nc104
        n=df104
    if opcion == 105:
        for i in range(len(c105)):
            if i == 0:
                nc105.append(c105[i])
            else:
                nc105.append(c105[i]-c105[i-1])
        df105 = pd.DataFrame()
        df105['Fecha'] = pd.to_datetime(f105)
        df105.index = df105['Fecha']
        df105['Casos'] = nc105
        n=df105
    if opcion == 106:
        for i in range(len(c106)):
            if i == 0:
                nc106.append(c106[i])
            else:
                nc106.append(c106[i]-c106[i-1])
        df106 = pd.DataFrame()
        df106['Fecha'] = pd.to_datetime(f106)
        df106.index = df106['Fecha']
        df106['Casos'] = nc106
        n=df106
    if opcion == 107:
        for i in range(len(c107)):
            if i == 0:
                nc107.append(c107[i])
            else:
                nc107.append(c107[i]-c107[i-1])
        df107 = pd.DataFrame()
        df107['Fecha'] = pd.to_datetime(f107)
        df107.index = df107['Fecha']
        df107['Casos'] = nc107
        n=df107
    if opcion == 108:
        for i in range(len(c108)):
            if i == 0:
                nc108.append(c108[i])
            else:
                nc108.append(c108[i]-c108[i-1])
        df108 = pd.DataFrame()
        df108['Fecha'] = pd.to_datetime(f108)
        df108.index = df108['Fecha']
        df108['Casos'] = nc108
        n=df108
    if opcion == 109:
        for i in range(len(c109)):
            if i == 0:
                nc109.append(c109[i])
            else:
                nc109.append(c109[i]-c109[i-1])
        df109 = pd.DataFrame()
        df109['Fecha'] = pd.to_datetime(f109)
        df109.index = df109['Fecha']
        df109['Casos'] = nc109
        n=df109
    if opcion == 110:
        for i in range(len(c110)):
            if i == 0:
                nc110.append(c110[i])
            else:
                nc110.append(c110[i]-c110[i-1])
        df110 = pd.DataFrame()
        df110['Fecha'] = pd.to_datetime(f110)
        df110.index = df110['Fecha']
        df110['Casos'] = nc110
        n=df110
    if opcion == 111:
        for i in range(len(c111)):
            if i == 0:
                nc111.append(c111[i])
            else:
                nc111.append(c111[i]-c111[i-1])
        df111 = pd.DataFrame()
        df111['Fecha'] = pd.to_datetime(f111)
        df111.index = df111['Fecha']
        df111['Casos'] = nc111
        n=df111
    if opcion == 112:
        for i in range(len(c112)):
            if i == 0:
                nc112.append(c112[i])
            else:
                nc112.append(c112[i]-c112[i-1])
        df112 = pd.DataFrame()
        df112['Fecha'] = pd.to_datetime(f112)
        df112.index = df112['Fecha']
        df112['Casos'] = nc112
        n=df112
    if opcion == 113:
        for i in range(len(c113)):
            if i == 0:
                nc113.append(c113[i])
            else:
                nc113.append(c113[i]-c113[i-1])
        df113 = pd.DataFrame()
        df113['Fecha'] = pd.to_datetime(f113)
        df113.index = df113['Fecha']
        df113['Casos'] = nc113
        n=df113
    if opcion == 114:
        for i in range(len(c114)):
            if i == 0:
                nc114.append(c114[i])
            else:
                nc114.append(c114[i]-c114[i-1])
        df114 = pd.DataFrame()
        df114['Fecha'] = pd.to_datetime(f114)
        df114.index = df114['Fecha']
        df114['Casos'] = nc114
        n=df114
    if opcion == 115:
        for i in range(len(c115)):
            if i == 0:
                nc115.append(c115[i])
            else:
                nc115.append(c115[i]-c115[i-1])
        df115 = pd.DataFrame()
        df115['Fecha'] = pd.to_datetime(f115)
        df115.index = df115['Fecha']
        df115['Casos'] = nc115
        n=df115
    if opcion == 116:
        for i in range(len(c116)):
            if i == 0:
                nc116.append(c116[i])
            else:
                nc116.append(c116[i]-c116[i-1])
        df116 = pd.DataFrame()
        df116['Fecha'] = pd.to_datetime(f116)
        df116.index = df116['Fecha']
        df116['Casos'] = nc116
        n=df116
    if opcion == 117:
        for i in range(len(c117)):
            if i == 0:
                nc117.append(c117[i])
            else:
                nc117.append(c117[i]-c117[i-1])
        df117 = pd.DataFrame()
        df117['Fecha'] = pd.to_datetime(f117)
        df117.index = df117['Fecha']
        df117['Casos'] = nc117
        n=df117
    if opcion == 118:
        for i in range(len(c118)):
            if i == 0:
                nc118.append(c118[i])
            else:
                nc118.append(c118[i]-c118[i-1])
        df118 = pd.DataFrame()
        df118['Fecha'] = pd.to_datetime(f118)
        df118.index = df118['Fecha']
        df118['Casos'] = nc118
        n=df118
    if opcion == 119:
        for i in range(len(c119)):
            if i == 0:
                nc119.append(c119[i])
            else:
                nc119.append(c119[i]-c119[i-1])
        df119 = pd.DataFrame()
        df119['Fecha'] = pd.to_datetime(f119)
        df119.index = df119['Fecha']
        df119['Casos'] = nc119
        n=df119
    if opcion == 120:
        for i in range(len(c120)):
            if i == 0:
                nc120.append(c120[i])
            else:
                nc120.append(c120[i]-c120[i-1])
        df120 = pd.DataFrame()
        df120['Fecha'] = pd.to_datetime(f120)
        df120.index = df120['Fecha']
        df120['Casos'] = nc120
        n=df120
    if opcion == 121:
        for i in range(len(c121)):
            if i == 0:
                nc121.append(c121[i])
            else:
                nc121.append(c121[i]-c121[i-1])
        df121 = pd.DataFrame()
        df121['Fecha'] = pd.to_datetime(f121)
        df121.index = df121['Fecha']
        df121['Casos'] = nc121
        n=df121
    if opcion == 122:
        for i in range(len(c122)):
            if i == 0:
                nc122.append(c122[i])
            else:
                nc122.append(c122[i]-c122[i-1])
        df122 = pd.DataFrame()
        df122['Fecha'] = pd.to_datetime(f122)
        df122.index = df122['Fecha']
        df122['Casos'] = nc122
        n=df122
    if opcion == 123:
        for i in range(len(c123)):
            if i == 0:
                nc123.append(c123[i])
            else:
                nc123.append(c123[i]-c123[i-1])
        df123 = pd.DataFrame()
        df123['Fecha'] = pd.to_datetime(f123)
        df123.index = df123['Fecha']
        df123['Casos'] = nc123
        n=df123
    if opcion == 124:
        for i in range(len(c124)):
            if i == 0:
                nc124.append(c124[i])
            else:
                nc124.append(c124[i]-c124[i-1])
        df124 = pd.DataFrame()
        df124['Fecha'] = pd.to_datetime(f124)
        df124.index = df124['Fecha']
        df124['Casos'] = nc124
        n=df124
    if opcion == 125:
        for i in range(len(c125)):
            if i == 0:
                nc125.append(c125[i])
            else:
                nc125.append(c125[i]-c125[i-1])
        df125 = pd.DataFrame()
        df125['Fecha'] = pd.to_datetime(f125)
        df125.index = df125['Fecha']
        df125['Casos'] = nc125
        n=df125
    if opcion == 126:
        for i in range(len(c126)):
            if i == 0:
                nc126.append(c126[i])
            else:
                nc126.append(c126[i]-c126[i-1])
        df126 = pd.DataFrame()
        df126['Fecha'] = pd.to_datetime(f126)
        df126.index = df126['Fecha']
        df126['Casos'] = nc126
        n=df126
    if opcion == 127:
        for i in range(len(c127)):
            if i == 0:
                nc127.append(c127[i])
            else:
                nc127.append(c127[i]-c127[i-1])
        df127 = pd.DataFrame()
        df127['Fecha'] = pd.to_datetime(f127)
        df127.index = df127['Fecha']
        df127['Casos'] = nc127
        n=df127
    if opcion == 128:
        for i in range(len(c128)):
            if i == 0:
                nc128.append(c128[i])
            else:
                nc128.append(c128[i]-c128[i-1])
        df128 = pd.DataFrame()
        df128['Fecha'] = pd.to_datetime(f128)
        df128.index = df128['Fecha']
        df128['Casos'] = nc128
        n=df128
    if opcion == 129:
        for i in range(len(c129)):
            if i == 0:
                nc129.append(c129[i])
            else:
                nc129.append(c129[i]-c129[i-1])
        df129 = pd.DataFrame()
        df129['Fecha'] = pd.to_datetime(f129)
        df129.index = df129['Fecha']
        df129['Casos'] = nc129
        n=df129
    if opcion == 130:
        for i in range(len(c130)):
            if i == 0:
                nc130.append(c130[i])
            else:
                nc130.append(c130[i]-c130[i-1])
        df130 = pd.DataFrame()
        df130['Fecha'] = pd.to_datetime(f130)
        df130.index = df130['Fecha']
        df130['Casos'] = nc130
        n=df130
    if opcion == 131:
        for i in range(len(c131)):
            if i == 0:
                nc131.append(c131[i])
            else:
                nc131.append(c131[i]-c131[i-1])
        df131 = pd.DataFrame()
        df131['Fecha'] = pd.to_datetime(f131)
        df131.index = df131['Fecha']
        df131['Casos'] = nc131
        n=df131
    if opcion == 132:
        for i in range(len(c132)):
            if i == 0:
                nc132.append(c132[i])
            else:
                nc132.append(c132[i]-c132[i-1])
        df132 = pd.DataFrame()
        df132['Fecha'] = pd.to_datetime(f132)
        df132.index = df132['Fecha']
        df132['Casos'] = nc132
        n=df132
    if opcion == 133:
        for i in range(len(c133)):
            if i == 0:
                nc133.append(c133[i])
            else:
                nc133.append(c133[i]-c133[i-1])
        df133 = pd.DataFrame()
        df133['Fecha'] = pd.to_datetime(f133)
        df133.index = df133['Fecha']
        df133['Casos'] = nc133
        n=df133
    if opcion == 134:
        for i in range(len(c134)):
            if i == 0:
                nc134.append(c134[i])
            else:
                nc134.append(c134[i]-c134[i-1])
        df134 = pd.DataFrame()
        df134['Fecha'] = pd.to_datetime(f134)
        df134.index = df134['Fecha']
        df134['Casos'] = nc134
        n=df134
    if opcion == 135:
        for i in range(len(c135)):
            if i == 0:
                nc135.append(c135[i])
            else:
                nc135.append(c135[i]-c135[i-1])
        df135 = pd.DataFrame()
        df135['Fecha'] = pd.to_datetime(f135)
        df135.index = df135['Fecha']
        df135['Casos'] = nc135
        n=df135
    if opcion == 136:
        for i in range(len(c136)):
            if i == 0:
                nc136.append(c136[i])
            else:
                nc136.append(c136[i]-c136[i-1])
        df136 = pd.DataFrame()
        df136['Fecha'] = pd.to_datetime(f136)
        df136.index = df136['Fecha']
        df136['Casos'] = nc136
        n=df136
    if opcion == 137:
        for i in range(len(c137)):
            if i == 0:
                nc137.append(c137[i])
            else:
                nc137.append(c137[i]-c137[i-1])
        df137 = pd.DataFrame()
        df137['Fecha'] = pd.to_datetime(f137)
        df137.index = df137['Fecha']
        df137['Casos'] = nc137
        n=df137
    if opcion == 138:
        for i in range(len(c138)):
            if i == 0:
                nc138.append(c138[i])
            else:
                nc138.append(c138[i]-c138[i-1])
        df138 = pd.DataFrame()
        df138['Fecha'] = pd.to_datetime(f138)
        df138.index = df138['Fecha']
        df138['Casos'] = nc138
        n=df138
    if opcion == 139:
        for i in range(len(c139)):
            if i == 0:
                nc139.append(c139[i])
            else:
                nc139.append(c139[i]-c139[i-1])
        df139 = pd.DataFrame()
        df139['Fecha'] = pd.to_datetime(f139)
        df139.index = df139['Fecha']
        df139['Casos'] = nc139
        n=df139
    if opcion == 140:
        for i in range(len(c140)):
            if i == 0:
                nc140.append(c140[i])
            else:
                nc140.append(c140[i]-c140[i-1])
        df140 = pd.DataFrame()
        df140['Fecha'] = pd.to_datetime(f140)
        df140.index = df140['Fecha']
        df140['Casos'] = nc140
        n=df140
    if opcion == 141:
        for i in range(len(c141)):
            if i == 0:
                nc141.append(c141[i])
            else:
                nc141.append(c141[i]-c141[i-1])
        df141 = pd.DataFrame()
        df141['Fecha'] = pd.to_datetime(f141)
        df141.index = df141['Fecha']
        df141['Casos'] = nc141
        n=df141
    if opcion == 142:
        for i in range(len(c142)):
            if i == 0:
                nc142.append(c142[i])
            else:
                nc142.append(c142[i]-c142[i-1])
        df142 = pd.DataFrame()
        df142['Fecha'] = pd.to_datetime(f142)
        df142.index = df142['Fecha']
        df142['Casos'] = nc142
        n=df142
    if opcion == 143:
        for i in range(len(c143)):
            if i == 0:
                nc143.append(c143[i])
            else:
                nc143.append(c143[i]-c143[i-1])
        df143 = pd.DataFrame()
        df143['Fecha'] = pd.to_datetime(f143)
        df143.index = df143['Fecha']
        df143['Casos'] = nc143
        n=df143
    if opcion == 144:
        for i in range(len(c144)):
            if i == 0:
                nc144.append(c144[i])
            else:
                nc144.append(c144[i]-c144[i-1])
        df144 = pd.DataFrame()
        df144['Fecha'] = pd.to_datetime(f144)
        df144.index = df144['Fecha']
        df144['Casos'] = nc144
        n=df144
    if opcion == 145:
        for i in range(len(c145)):
            if i == 0:
                nc145.append(c145[i])
            else:
                nc145.append(c145[i]-c145[i-1])
        df145 = pd.DataFrame()
        df145['Fecha'] = pd.to_datetime(f145)
        df145.index = df145['Fecha']
        df145['Casos'] = nc145
        n=df145
    if opcion == 146:
        for i in range(len(c146)):
            if i == 0:
                nc146.append(c146[i])
            else:
                nc146.append(c146[i]-c146[i-1])
        df146 = pd.DataFrame()
        df146['Fecha'] = pd.to_datetime(f146)
        df146.index = df146['Fecha']
        df146['Casos'] = nc146
        n=df146
    if opcion == 147:
        for i in range(len(c147)):
            if i == 0:
                nc147.append(c147[i])
            else:
                nc147.append(c147[i]-c147[i-1])
        df147 = pd.DataFrame()
        df147['Fecha'] = pd.to_datetime(f147)
        df147.index = df147['Fecha']
        df147['Casos'] = nc147
        n=df147
    if opcion == 148:
        for i in range(len(c148)):
            if i == 0:
                nc148.append(c148[i])
            else:
                nc148.append(c148[i]-c148[i-1])
        df148 = pd.DataFrame()
        df148['Fecha'] = pd.to_datetime(f148)
        df148.index = df148['Fecha']
        df148['Casos'] = nc148
        n=df148
    if opcion == 149:
        for i in range(len(c149)):
            if i == 0:
                nc149.append(c149[i])
            else:
                nc149.append(c149[i]-c149[i-1])
        df149 = pd.DataFrame()
        df149['Fecha'] = pd.to_datetime(f149)
        df149.index = df149['Fecha']
        df149['Casos'] = nc149
        n=df149
    if opcion == 150:
        for i in range(len(c150)):
            if i == 0:
                nc150.append(c150[i])
            else:
                nc150.append(c150[i]-c150[i-1])
        df150 = pd.DataFrame()
        df150['Fecha'] = pd.to_datetime(f150)
        df150.index = df150['Fecha']
        df150['Casos'] = nc150
        n=df150
    if opcion == 151:
        for i in range(len(c151)):
            if i == 0:
                nc151.append(c151[i])
            else:
                nc151.append(c151[i]-c151[i-1])
        df151 = pd.DataFrame()
        df151['Fecha'] = pd.to_datetime(f151)
        df151.index = df151['Fecha']
        df151['Casos'] = nc151
        n=df151
    if opcion == 152:
        for i in range(len(c152)):
            if i == 0:
                nc152.append(c152[i])
            else:
                nc152.append(c152[i]-c152[i-1])
        df152 = pd.DataFrame()
        df152['Fecha'] = pd.to_datetime(f152)
        df152.index = df152['Fecha']
        df152['Casos'] = nc152
        n=df152
    if opcion == 153:
        for i in range(len(c153)):
            if i == 0:
                nc153.append(c153[i])
            else:
                nc153.append(c153[i]-c153[i-1])
        df153 = pd.DataFrame()
        df153['Fecha'] = pd.to_datetime(f153)
        df153.index = df153['Fecha']
        df153['Casos'] = nc153
        n=df153
    if opcion == 154:
        for i in range(len(c154)):
            if i == 0:
                nc154.append(c154[i])
            else:
                nc154.append(c154[i]-c154[i-1])
        df154 = pd.DataFrame()
        df154['Fecha'] = pd.to_datetime(f154)
        df154.index = df154['Fecha']
        df154['Casos'] = nc154
        n=df154
    if opcion == 155:
        for i in range(len(c155)):
            if i == 0:
                nc155.append(c155[i])
            else:
                nc155.append(c155[i]-c155[i-1])
        df155 = pd.DataFrame()
        df155['Fecha'] = pd.to_datetime(f155)
        df155.index = df155['Fecha']
        df155['Casos'] = nc155
        n=df155
    if opcion == 156:
        for i in range(len(c156)):
            if i == 0:
                nc156.append(c156[i])
            else:
                nc156.append(c156[i]-c156[i-1])
        df156 = pd.DataFrame()
        df156['Fecha'] = pd.to_datetime(f156)
        df156.index = df156['Fecha']
        df156['Casos'] = nc156
        n=df156
    if opcion == 157:
        for i in range(len(c157)):
            if i == 0:
                nc157.append(c157[i])
            else:
                nc157.append(c157[i]-c157[i-1])
        df157 = pd.DataFrame()
        df157['Fecha'] = pd.to_datetime(f157)
        df157.index = df157['Fecha']
        df157['Casos'] = nc157
        n=df157
    if opcion == 158:
        for i in range(len(c158)):
            if i == 0:
                nc158.append(c158[i])
            else:
                nc158.append(c158[i]-c158[i-1])
        df158 = pd.DataFrame()
        df158['Fecha'] = pd.to_datetime(f158)
        df158.index = df158['Fecha']
        df158['Casos'] = nc158
        n=df158
    if opcion == 159:
        for i in range(len(c159)):
            if i == 0:
                nc159.append(c159[i])
            else:
                nc159.append(c159[i]-c159[i-1])
        df159 = pd.DataFrame()
        df159['Fecha'] = pd.to_datetime(f159)
        df159.index = df159['Fecha']
        df159['Casos'] = nc159
        n=df159
    if opcion == 160:
        for i in range(len(c160)):
            if i == 0:
                nc160.append(c160[i])
            else:
                nc160.append(c160[i]-c160[i-1])
        df160 = pd.DataFrame()
        df160['Fecha'] = pd.to_datetime(f160)
        df160.index = df160['Fecha']
        df160['Casos'] = nc160
        n=df160
    if opcion == 161:
        for i in range(len(c161)):
            if i == 0:
                nc161.append(c161[i])
            else:
                nc161.append(c161[i]-c161[i-1])
        df161 = pd.DataFrame()
        df161['Fecha'] = pd.to_datetime(f161)
        df161.index = df161['Fecha']
        df161['Casos'] = nc161
        n=df161
    if opcion == 162:
        for i in range(len(c162)):
            if i == 0:
                nc162.append(c162[i])
            else:
                nc162.append(c162[i]-c162[i-1])
        df162 = pd.DataFrame()
        df162['Fecha'] = pd.to_datetime(f162)
        df162.index = df162['Fecha']
        df162['Casos'] = nc162
        n=df162
    if opcion == 163:
        for i in range(len(c163)):
            if i == 0:
                nc163.append(c163[i])
            else:
                nc163.append(c163[i]-c163[i-1])
        df163 = pd.DataFrame()
        df163['Fecha'] = pd.to_datetime(f163)
        df163.index = df163['Fecha']
        df163['Casos'] = nc163
        n=df163
    if opcion == 164:
        for i in range(len(c164)):
            if i == 0:
                nc164.append(c164[i])
            else:
                nc164.append(c164[i]-c164[i-1])
        df164 = pd.DataFrame()
        df164['Fecha'] = pd.to_datetime(f164)
        df164.index = df164['Fecha']
        df164['Casos'] = nc164
        n=df164
    if opcion == 165:
        for i in range(len(c165)):
            if i == 0:
                nc165.append(c165[i])
            else:
                nc165.append(c165[i]-c165[i-1])
        df165 = pd.DataFrame()
        df165['Fecha'] = pd.to_datetime(f165)
        df165.index = df165['Fecha']
        df165['Casos'] = nc165
        n=df165
    if opcion == 166:
        for i in range(len(c166)):
            if i == 0:
                nc166.append(c166[i])
            else:
                nc166.append(c166[i]-c166[i-1])
        df166 = pd.DataFrame()
        df166['Fecha'] = pd.to_datetime(f166)
        df166.index = df166['Fecha']
        df166['Casos'] = nc166
        n=df166
    if opcion == 167:
        for i in range(len(c167)):
            if i == 0:
                nc167.append(c167[i])
            else:
                nc167.append(c167[i]-c167[i-1])
        df167 = pd.DataFrame()
        df167['Fecha'] = pd.to_datetime(f167)
        df167.index = df167['Fecha']
        df167['Casos'] = nc167
        n=df167
    if opcion == 168:
        for i in range(len(c168)):
            if i == 0:
                nc168.append(c168[i])
            else:
                nc168.append(c168[i]-c168[i-1])
        df168 = pd.DataFrame()
        df168['Fecha'] = pd.to_datetime(f168)
        df168.index = df168['Fecha']
        df168['Casos'] = nc168
        n=df168
    if opcion == 169:
        for i in range(len(c169)):
            if i == 0:
                nc169.append(c169[i])
            else:
                nc169.append(c169[i]-c169[i-1])
        df169 = pd.DataFrame()
        df169['Fecha'] = pd.to_datetime(f169)
        df169.index = df169['Fecha']
        df169['Casos'] = nc169
        n=df169
    if opcion == 170:
        for i in range(len(c170)):
            if i == 0:
                nc170.append(c170[i])
            else:
                nc170.append(c170[i]-c170[i-1])
        df170 = pd.DataFrame()
        df170['Fecha'] = pd.to_datetime(f170)
        df170.index = df170['Fecha']
        df170['Casos'] = nc170
        n=df170
    if opcion == 171:
        for i in range(len(c171)):
            if i == 0:
                nc171.append(c171[i])
            else:
                nc171.append(c171[i]-c171[i-1])
        df171 = pd.DataFrame()
        df171['Fecha'] = pd.to_datetime(f171)
        df171.index = df171['Fecha']
        df171['Casos'] = nc171
        n=df171
    if opcion == 172:
        for i in range(len(c172)):
            if i == 0:
                nc172.append(c172[i])
            else:
                nc172.append(c172[i]-c172[i-1])
        df172 = pd.DataFrame()
        df172['Fecha'] = pd.to_datetime(f172)
        df172.index = df172['Fecha']
        df172['Casos'] = nc172
        n=df172
    if opcion == 173:
        for i in range(len(c173)):
            if i == 0:
                nc173.append(c173[i])
            else:
                nc173.append(c173[i]-c173[i-1])
        df173 = pd.DataFrame()
        df173['Fecha'] = pd.to_datetime(f173)
        df173.index = df173['Fecha']
        df173['Casos'] = nc173
        n=df173
    if opcion == 174:
        for i in range(len(c174)):
            if i == 0:
                nc174.append(c174[i])
            else:
                nc174.append(c174[i]-c174[i-1])
        df174 = pd.DataFrame()
        df174['Fecha'] = pd.to_datetime(f174)
        df174.index = df174['Fecha']
        df174['Casos'] = nc174
        n=df174
    if opcion == 175:
        for i in range(len(c175)):
            if i == 0:
                nc175.append(c175[i])
            else:
                nc175.append(c175[i]-c175[i-1])
        df175 = pd.DataFrame()
        df175['Fecha'] = pd.to_datetime(f175)
        df175.index = df175['Fecha']
        df175['Casos'] = nc175
        n=df175
    if opcion == 176:
        for i in range(len(c176)):
            if i == 0:
                nc176.append(c176[i])
            else:
                nc176.append(c176[i]-c176[i-1])
        df176 = pd.DataFrame()
        df176['Fecha'] = pd.to_datetime(f176)
        df176.index = df176['Fecha']
        df176['Casos'] = nc176
        n=df176
    if opcion == 177:
        for i in range(len(c177)):
            if i == 0:
                nc177.append(c177[i])
            else:
                nc177.append(c177[i]-c177[i-1])
        df177 = pd.DataFrame()
        df177['Fecha'] = pd.to_datetime(f177)
        df177.index = df177['Fecha']
        df177['Casos'] = nc177
        n=df177
    if opcion == 178:
        for i in range(len(c178)):
            if i == 0:
                nc178.append(c178[i])
            else:
                nc178.append(c178[i]-c178[i-1])
        df178 = pd.DataFrame()
        df178['Fecha'] = pd.to_datetime(f178)
        df178.index = df178['Fecha']
        df178['Casos'] = nc178
        n=df178
    if opcion == 179:
        for i in range(len(c179)):
            if i == 0:
                nc179.append(c179[i])
            else:
                nc179.append(c179[i]-c179[i-1])
        df179 = pd.DataFrame()
        df179['Fecha'] = pd.to_datetime(f179)
        df179.index = df179['Fecha']
        df179['Casos'] = nc179
        n=df179
    if opcion == 180:
        for i in range(len(c180)):
            if i == 0:
                nc180.append(c180[i])
            else:
                nc180.append(c180[i]-c180[i-1])
        df180 = pd.DataFrame()
        df180['Fecha'] = pd.to_datetime(f180)
        df180.index = df180['Fecha']
        df180['Casos'] = nc180
        n=df180
    if opcion == 181:
        for i in range(len(c181)):
            if i == 0:
                nc181.append(c181[i])
            else:
                nc181.append(c181[i]-c181[i-1])
        df181 = pd.DataFrame()
        df181['Fecha'] = pd.to_datetime(f181)
        df181.index = df181['Fecha']
        df181['Casos'] = nc181
        n=df181
    if opcion == 182:
        for i in range(len(c182)):
            if i == 0:
                nc182.append(c182[i])
            else:
                nc182.append(c182[i]-c182[i-1])
        df182 = pd.DataFrame()
        df182['Fecha'] = pd.to_datetime(f182)
        df182.index = df182['Fecha']
        df182['Casos'] = nc182
        n=df182
    if opcion == 183:
        for i in range(len(c183)):
            if i == 0:
                nc183.append(c183[i])
            else:
                nc183.append(c183[i]-c183[i-1])
        df183 = pd.DataFrame()
        df183['Fecha'] = pd.to_datetime(f183)
        df183.index = df183['Fecha']
        df183['Casos'] = nc183
        n=df183
    if opcion == 184:
        for i in range(len(c184)):
            if i == 0:
                nc184.append(c184[i])
            else:
                nc184.append(c184[i]-c184[i-1])
        df184 = pd.DataFrame()
        df184['Fecha'] = pd.to_datetime(f184)
        df184.index = df184['Fecha']
        df184['Casos'] = nc184
        n=df184
    if opcion == 185:
        for i in range(len(c185)):
            if i == 0:
                nc185.append(c185[i])
            else:
                nc185.append(c185[i]-c185[i-1])
        df185 = pd.DataFrame()
        df185['Fecha'] = pd.to_datetime(f185)
        df185.index = df185['Fecha']
        df185['Casos'] = nc185
        n=df185
    if opcion == 186:
        for i in range(len(c186)):
            if i == 0:
                nc186.append(c186[i])
            else:
                nc186.append(c186[i]-c186[i-1])
        df186 = pd.DataFrame()
        df186['Fecha'] = pd.to_datetime(f186)
        df186.index = df186['Fecha']
        df186['Casos'] = nc186
        n=df186
    if opcion == 187:
        for i in range(len(c187)):
            if i == 0:
                nc187.append(c187[i])
            else:
                nc187.append(c187[i]-c187[i-1])
        df187 = pd.DataFrame()
        df187['Fecha'] = pd.to_datetime(f187)
        df187.index = df187['Fecha']
        df187['Casos'] = nc187
        n=df187
    if opcion == 188:
        for i in range(len(c188)):
            if i == 0:
                nc188.append(c188[i])
            else:
                nc188.append(c188[i]-c188[i-1])
        df188 = pd.DataFrame()
        df188['Fecha'] = pd.to_datetime(f188)
        df188.index = df188['Fecha']
        df188['Casos'] = nc188
        n=df188
    if opcion == 189:
        for i in range(len(c189)):
            if i == 0:
                nc189.append(c189[i])
            else:
                nc189.append(c189[i]-c189[i-1])
        df189 = pd.DataFrame()
        df189['Fecha'] = pd.to_datetime(f189)
        df189.index = df189['Fecha']
        df189['Casos'] = nc189
        n=df189
    if opcion == 190:
        for i in range(len(c190)):
            if i == 0:
                nc190.append(c190[i])
            else:
                nc190.append(c190[i]-c190[i-1])
        df190 = pd.DataFrame()
        df190['Fecha'] = pd.to_datetime(f190)
        df190.index = df190['Fecha']
        df190['Casos'] = nc190
        n=df190
    if opcion == 191:
        for i in range(len(c191)):
            if i == 0:
                nc191.append(c191[i])
            else:
                nc191.append(c191[i]-c191[i-1])
        df191 = pd.DataFrame()
        df191['Fecha'] = pd.to_datetime(f191)
        df191.index = df191['Fecha']
        df191['Casos'] = nc191
        n=df191
    if opcion == 192:
        for i in range(len(c192)):
            if i == 0:
                nc192.append(c192[i])
            else:
                nc192.append(c192[i]-c192[i-1])
        df192 = pd.DataFrame()
        df192['Fecha'] = pd.to_datetime(f192)
        df192.index = df192['Fecha']
        df192['Casos'] = nc192
        n=df192
    if opcion == 193:
        for i in range(len(c193)):
            if i == 0:
                nc193.append(c193[i])
            else:
                nc193.append(c193[i]-c193[i-1])
        df193 = pd.DataFrame()
        df193['Fecha'] = pd.to_datetime(f193)
        df193.index = df193['Fecha']
        df193['Casos'] = nc193
        n=df193
    if opcion == 194:
        for i in range(len(c194)):
            if i == 0:
                nc194.append(c194[i])
            else:
                nc194.append(c194[i]-c194[i-1])
        df194 = pd.DataFrame()
        df194['Fecha'] = pd.to_datetime(f194)
        df194.index = df194['Fecha']
        df194['Casos'] = nc194
        n=df194
    if opcion == 195:
        for i in range(len(c195)):
            if i == 0:
                nc195.append(c195[i])
            else:
                nc195.append(c195[i]-c195[i-1])
        df195 = pd.DataFrame()
        df195['Fecha'] = pd.to_datetime(f195)
        df195.index = df195['Fecha']
        df195['Casos'] = nc195
        n=df195
    if opcion == 196:
        for i in range(len(c196)):
            if i == 0:
                nc196.append(c196[i])
            else:
                nc196.append(c196[i]-c196[i-1])
        df196 = pd.DataFrame()
        df196['Fecha'] = pd.to_datetime(f196)
        df196.index = df196['Fecha']
        df196['Casos'] = nc196
        n=df196
    if opcion == 197:
        for i in range(len(c197)):
            if i == 0:
                nc197.append(c197[i])
            else:
                nc197.append(c197[i]-c197[i-1])
        df197 = pd.DataFrame()
        df197['Fecha'] = pd.to_datetime(f197)
        df197.index = df197['Fecha']
        df197['Casos'] = nc197
        n=df197
    if opcion == 198:
        for i in range(len(c198)):
            if i == 0:
                nc198.append(c198[i])
            else:
                nc198.append(c198[i]-c198[i-1])
        df198 = pd.DataFrame()
        df198['Fecha'] = pd.to_datetime(f198)
        df198.index = df198['Fecha']
        df198['Casos'] = nc198
        n=df198
    if opcion == 199:
        for i in range(len(c199)):
            if i == 0:
                nc199.append(c199[i])
            else:
                nc199.append(c199[i]-c199[i-1])
        df199 = pd.DataFrame()
        df199['Fecha'] = pd.to_datetime(f199)
        df199.index = df199['Fecha']
        df199['Casos'] = nc199
        n=df199
    if opcion == 200:
        for i in range(len(c200)):
            if i == 0:
                nc200.append(c200[i])
            else:
                nc200.append(c200[i]-c200[i-1])
        df200 = pd.DataFrame()
        df200['Fecha'] = pd.to_datetime(f200)
        df200.index = df200['Fecha']
        df200['Casos'] = nc200
        n=df200
    if opcion == 201:
        for i in range(len(c201)):
            if i == 0:
                nc201.append(c201[i])
            else:
                nc201.append(c201[i]-c201[i-1])
        df201 = pd.DataFrame()
        df201['Fecha'] = pd.to_datetime(f201)
        df201.index = df201['Fecha']
        df201['Casos'] = nc201
        n=df201
    if opcion == 202:
        for i in range(len(c202)):
            if i == 0:
                nc202.append(c202[i])
            else:
                nc202.append(c202[i]-c202[i-1])
        df202 = pd.DataFrame()
        df202['Fecha'] = pd.to_datetime(f202)
        df202.index = df202['Fecha']
        df202['Casos'] = nc202
        n=df202
    if opcion == 203:
        for i in range(len(c203)):
            if i == 0:
                nc203.append(c203[i])
            else:
                nc203.append(c203[i]-c203[i-1])
        df203 = pd.DataFrame()
        df203['Fecha'] = pd.to_datetime(f203)
        df203.index = df203['Fecha']
        df203['Casos'] = nc203
        n=df203
    if opcion == 204:
        for i in range(len(c204)):
            if i == 0:
                nc204.append(c204[i])
            else:
                nc204.append(c204[i]-c204[i-1])
        df204 = pd.DataFrame()
        df204['Fecha'] = pd.to_datetime(f204)
        df204.index = df204['Fecha']
        df204['Casos'] = nc204
        n=df204
    if opcion == 205:
        for i in range(len(c205)):
            if i == 0:
                nc205.append(c205[i])
            else:
                nc205.append(c205[i]-c205[i-1])
        df205 = pd.DataFrame()
        df205['Fecha'] = pd.to_datetime(f205)
        df205.index = df205['Fecha']
        df205['Casos'] = nc205
        n=df205
    if opcion == 206:
        for i in range(len(c206)):
            if i == 0:
                nc206.append(c206[i])
            else:
                nc206.append(c206[i]-c206[i-1])
        df206 = pd.DataFrame()
        df206['Fecha'] = pd.to_datetime(f206)
        df206.index = df206['Fecha']
        df206['Casos'] = nc206
        n=df206
    if opcion == 207:
        for i in range(len(c207)):
            if i == 0:
                nc207.append(c207[i])
            else:
                nc207.append(c207[i]-c207[i-1])
        df207 = pd.DataFrame()
        df207['Fecha'] = pd.to_datetime(f207)
        df207.index = df207['Fecha']
        df207['Casos'] = nc207
        n=df207
    if opcion == 208:
        for i in range(len(c208)):
            if i == 0:
                nc208.append(c208[i])
            else:
                nc208.append(c208[i]-c208[i-1])
        df208 = pd.DataFrame()
        df208['Fecha'] = pd.to_datetime(f208)
        df208.index = df208['Fecha']
        df208['Casos'] = nc208
        n=df208
    if opcion == 209:
        for i in range(len(c209)):
            if i == 0:
                nc209.append(c209[i])
            else:
                nc209.append(c209[i]-c209[i-1])
        df209 = pd.DataFrame()
        df209['Fecha'] = pd.to_datetime(f209)
        df209.index = df209['Fecha']
        df209['Casos'] = nc209
        n=df209
    if opcion == 210:
        for i in range(len(c210)):
            if i == 0:
                nc210.append(c210[i])
            else:
                nc210.append(c210[i]-c210[i-1])
        df210 = pd.DataFrame()
        df210['Fecha'] = pd.to_datetime(f210)
        df210.index = df210['Fecha']
        df210['Casos'] = nc210
        n=df210
    if opcion == 211:
        for i in range(len(c211)):
            if i == 0:
                nc211.append(c211[i])
            else:
                nc211.append(c211[i]-c211[i-1])
        df211 = pd.DataFrame()
        df211['Fecha'] = pd.to_datetime(f211)
        df211.index = df211['Fecha']
        df211['Casos'] = nc211
        n=df211
    if opcion == 212:
        for i in range(len(c212)):
            if i == 0:
                nc212.append(c212[i])
            else:
                nc212.append(c212[i]-c212[i-1])
        df212 = pd.DataFrame()
        df212['Fecha'] = pd.to_datetime(f212)
        df212.index = df212['Fecha']
        df212['Casos'] = nc212
        n=df212
    if opcion == 213:
        for i in range(len(c213)):
            if i == 0:
                nc213.append(c213[i])
            else:
                nc213.append(c213[i]-c213[i-1])
        df213 = pd.DataFrame()
        df213['Fecha'] = pd.to_datetime(f213)
        df213.index = df213['Fecha']
        df213['Casos'] = nc213
        n=df213
    if opcion == 214:
        for i in range(len(c214)):
            if i == 0:
                nc214.append(c214[i])
            else:
                nc214.append(c214[i]-c214[i-1])
        df214 = pd.DataFrame()
        df214['Fecha'] = pd.to_datetime(f214)
        df214.index = df214['Fecha']
        df214['Casos'] = nc214
        n=df214
    if opcion == 215:
        for i in range(len(c215)):
            if i == 0:
                nc215.append(c215[i])
            else:
                nc215.append(c215[i]-c215[i-1])
        df215 = pd.DataFrame()
        df215['Fecha'] = pd.to_datetime(f215)
        df215.index = df215['Fecha']
        df215['Casos'] = nc215
        n=df215
    if opcion == 216:
        for i in range(len(c216)):
            if i == 0:
                nc216.append(c216[i])
            else:
                nc216.append(c216[i]-c216[i-1])
        df216 = pd.DataFrame()
        df216['Fecha'] = pd.to_datetime(f216)
        df216.index = df216['Fecha']
        df216['Casos'] = nc216
        n=df216
    if opcion == 217:
        for i in range(len(c217)):
            if i == 0:
                nc217.append(c217[i])
            else:
                nc217.append(c217[i]-c217[i-1])
        df217 = pd.DataFrame()
        df217['Fecha'] = pd.to_datetime(f217)
        df217.index = df217['Fecha']
        df217['Casos'] = nc217
        n=df217
    if opcion == 218:
        for i in range(len(c218)):
            if i == 0:
                nc218.append(c218[i])
            else:
                nc218.append(c218[i]-c218[i-1])
        df218 = pd.DataFrame()
        df218['Fecha'] = pd.to_datetime(f218)
        df218.index = df218['Fecha']
        df218['Casos'] = nc218
        n=df218
    if opcion == 219:
        for i in range(len(c219)):
            if i == 0:
                nc219.append(c219[i])
            else:
                nc219.append(c219[i]-c219[i-1])
        df219 = pd.DataFrame()
        df219['Fecha'] = pd.to_datetime(f219)
        df219.index = df219['Fecha']
        df219['Casos'] = nc219
        n=df219
    if opcion == 220:
        for i in range(len(c220)):
            if i == 0:
                nc220.append(c220[i])
            else:
                nc220.append(c220[i]-c220[i-1])
        df220 = pd.DataFrame()
        df220['Fecha'] = pd.to_datetime(f220)
        df220.index = df220['Fecha']
        df220['Casos'] = nc220
        n=df220
    if opcion == 221:
        for i in range(len(c221)):
            if i == 0:
                nc221.append(c221[i])
            else:
                nc221.append(c221[i]-c221[i-1])
        df221 = pd.DataFrame()
        df221['Fecha'] = pd.to_datetime(f221)
        df221.index = df221['Fecha']
        df221['Casos'] = nc221
        n=df221
    if opcion == 222:
        for i in range(len(c222)):
            if i == 0:
                nc222.append(c222[i])
            else:
                nc222.append(c222[i]-c222[i-1])
        df222 = pd.DataFrame()
        df222['Fecha'] = pd.to_datetime(f222)
        df222.index = df222['Fecha']
        df222['Casos'] = nc222
        n=df222
    if opcion == 223:
        for i in range(len(c223)):
            if i == 0:
                nc223.append(c223[i])
            else:
                nc223.append(c223[i]-c223[i-1])
        df223 = pd.DataFrame()
        df223['Fecha'] = pd.to_datetime(f223)
        df223.index = df223['Fecha']
        df223['Casos'] = nc223
        n=df223
    if opcion == 224:
        for i in range(len(c224)):
            if i == 0:
                nc224.append(c224[i])
            else:
                nc224.append(c224[i]-c224[i-1])
        df224 = pd.DataFrame()
        df224['Fecha'] = pd.to_datetime(f224)
        df224.index = df224['Fecha']
        df224['Casos'] = nc224
        n=df224
    if opcion == 225:
        for i in range(len(c225)):
            if i == 0:
                nc225.append(c225[i])
            else:
                nc225.append(c225[i]-c225[i-1])
        df225 = pd.DataFrame()
        df225['Fecha'] = pd.to_datetime(f225)
        df225.index = df225['Fecha']
        df225['Casos'] = nc225
        n=df225
    if opcion == 226:
        for i in range(len(c226)):
            if i == 0:
                nc226.append(c226[i])
            else:
                nc226.append(c226[i]-c226[i-1])
        df226 = pd.DataFrame()
        df226['Fecha'] = pd.to_datetime(f226)
        df226.index = df226['Fecha']
        df226['Casos'] = nc226
        n=df226
    if opcion == 227:
        for i in range(len(c227)):
            if i == 0:
                nc227.append(c227[i])
            else:
                nc227.append(c227[i]-c227[i-1])
        df227 = pd.DataFrame()
        df227['Fecha'] = pd.to_datetime(f227)
        df227.index = df227['Fecha']
        df227['Casos'] = nc227
        n=df227
    if opcion == 228:
        for i in range(len(c228)):
            if i == 0:
                nc228.append(c228[i])
            else:
                nc228.append(c228[i]-c228[i-1])
        df228 = pd.DataFrame()
        df228['Fecha'] = pd.to_datetime(f228)
        df228.index = df228['Fecha']
        df228['Casos'] = nc228
        n=df228
    if opcion == 229:
        for i in range(len(c229)):
            if i == 0:
                nc229.append(c229[i])
            else:
                nc229.append(c229[i]-c229[i-1])
        df229 = pd.DataFrame()
        df229['Fecha'] = pd.to_datetime(f229)
        df229.index = df229['Fecha']
        df229['Casos'] = nc229
        n=df229
    if opcion == 230:
        for i in range(len(c230)):
            if i == 0:
                nc230.append(c230[i])
            else:
                nc230.append(c230[i]-c230[i-1])
        df230 = pd.DataFrame()
        df230['Fecha'] = pd.to_datetime(f230)
        df230.index = df230['Fecha']
        df230['Casos'] = nc230
        n=df230
    if opcion == 231:
        for i in range(len(c231)):
            if i == 0:
                nc231.append(c231[i])
            else:
                nc231.append(c231[i]-c231[i-1])
        df231 = pd.DataFrame()
        df231['Fecha'] = pd.to_datetime(f231)
        df231.index = df231['Fecha']
        df231['Casos'] = nc231
        n=df231
    if opcion == 232:
        for i in range(len(c232)):
            if i == 0:
                nc232.append(c232[i])
            else:
                nc232.append(c232[i]-c232[i-1])
        df232 = pd.DataFrame()
        df232['Fecha'] = pd.to_datetime(f232)
        df232.index = df232['Fecha']
        df232['Casos'] = nc232
        n=df232
    if opcion == 233:
        for i in range(len(c233)):
            if i == 0:
                nc233.append(c233[i])
            else:
                nc233.append(c233[i]-c233[i-1])
        df233 = pd.DataFrame()
        df233['Fecha'] = pd.to_datetime(f233)
        df233.index = df233['Fecha']
        df233['Casos'] = nc233
        n=df233
    if opcion == 234:
        for i in range(len(c234)):
            if i == 0:
                nc234.append(c234[i])
            else:
                nc234.append(c234[i]-c234[i-1])
        df234 = pd.DataFrame()
        df234['Fecha'] = pd.to_datetime(f234)
        df234.index = df234['Fecha']
        df234['Casos'] = nc234
        n=df234
    if opcion == 235:
        for i in range(len(c235)):
            if i == 0:
                nc235.append(c235[i])
            else:
                nc235.append(c235[i]-c235[i-1])
        df235 = pd.DataFrame()
        df235['Fecha'] = pd.to_datetime(f235)
        df235.index = df235['Fecha']
        df235['Casos'] = nc235
        n=df235
    if opcion == 236:
        for i in range(len(c236)):
            if i == 0:
                nc236.append(c236[i])
            else:
                nc236.append(c236[i]-c236[i-1])
        df236 = pd.DataFrame()
        df236['Fecha'] = pd.to_datetime(f236)
        df236.index = df236['Fecha']
        df236['Casos'] = nc236
        n=df236
    if opcion == 237:
        for i in range(len(c237)):
            if i == 0:
                nc237.append(c237[i])
            else:
                nc237.append(c237[i]-c237[i-1])
        df237 = pd.DataFrame()
        df237['Fecha'] = pd.to_datetime(f237)
        df237.index = df237['Fecha']
        df237['Casos'] = nc237
        n=df237
    if opcion == 238:
        for i in range(len(c238)):
            if i == 0:
                nc238.append(c238[i])
            else:
                nc238.append(c238[i]-c238[i-1])
        df238 = pd.DataFrame()
        df238['Fecha'] = pd.to_datetime(f238)
        df238.index = df238['Fecha']
        df238['Casos'] = nc238
        n=df238
    if opcion == 239:
        for i in range(len(c239)):
            if i == 0:
                nc239.append(c239[i])
            else:
                nc239.append(c239[i]-c239[i-1])
        df239 = pd.DataFrame()
        df239['Fecha'] = pd.to_datetime(f239)
        df239.index = df239['Fecha']
        df239['Casos'] = nc239
        n=df239
    if opcion == 240:
        for i in range(len(c240)):
            if i == 0:
                nc240.append(c240[i])
            else:
                nc240.append(c240[i]-c240[i-1])
        df240 = pd.DataFrame()
        df240['Fecha'] = pd.to_datetime(f240)
        df240.index = df240['Fecha']
        df240['Casos'] = nc240
        n=df240
    if opcion == 241:
        for i in range(len(c241)):
            if i == 0:
                nc241.append(c241[i])
            else:
                nc241.append(c241[i]-c241[i-1])
        df241 = pd.DataFrame()
        df241['Fecha'] = pd.to_datetime(f241)
        df241.index = df241['Fecha']
        df241['Casos'] = nc241
        n=df241
    if opcion == 242:
        for i in range(len(c242)):
            if i == 0:
                nc242.append(c242[i])
            else:
                nc242.append(c242[i]-c242[i-1])
        df242 = pd.DataFrame()
        df242['Fecha'] = pd.to_datetime(f242)
        df242.index = df242['Fecha']
        df242['Casos'] = nc242
        n=df242
    if opcion == 243:
        for i in range(len(c243)):
            if i == 0:
                nc243.append(c243[i])
            else:
                nc243.append(c243[i]-c243[i-1])
        df243 = pd.DataFrame()
        df243['Fecha'] = pd.to_datetime(f243)
        df243.index = df243['Fecha']
        df243['Casos'] = nc243
        n=df243
    if opcion == 244:
        for i in range(len(c244)):
            if i == 0:
                nc244.append(c244[i])
            else:
                nc244.append(c244[i]-c244[i-1])
        df244 = pd.DataFrame()
        df244['Fecha'] = pd.to_datetime(f244)
        df244.index = df244['Fecha']
        df244['Casos'] = nc244
        n=df244
    if opcion == 245:
        for i in range(len(c245)):
            if i == 0:
                nc245.append(c245[i])
            else:
                nc245.append(c245[i]-c245[i-1])
        df245 = pd.DataFrame()
        df245['Fecha'] = pd.to_datetime(f245)
        df245.index = df245['Fecha']
        df245['Casos'] = nc245
        n=df245
    if opcion == 246:
        for i in range(len(c246)):
            if i == 0:
                nc246.append(c246[i])
            else:
                nc246.append(c246[i]-c246[i-1])
        df246 = pd.DataFrame()
        df246['Fecha'] = pd.to_datetime(f246)
        df246.index = df246['Fecha']
        df246['Casos'] = nc246
        n=df246
    if opcion == 247:
        for i in range(len(c247)):
            if i == 0:
                nc247.append(c247[i])
            else:
                nc247.append(c247[i]-c247[i-1])
        df247 = pd.DataFrame()
        df247['Fecha'] = pd.to_datetime(f247)
        df247.index = df247['Fecha']
        df247['Casos'] = nc247
        n=df247
    if opcion == 248:
        for i in range(len(c248)):
            if i == 0:
                nc248.append(c248[i])
            else:
                nc248.append(c248[i]-c248[i-1])
        df248 = pd.DataFrame()
        df248['Fecha'] = pd.to_datetime(f248)
        df248.index = df248['Fecha']
        df248['Casos'] = nc248
        n=df248
    if opcion == 249:
        for i in range(len(c249)):
            if i == 0:
                nc249.append(c249[i])
            else:
                nc249.append(c249[i]-c249[i-1])
        df249 = pd.DataFrame()
        df249['Fecha'] = pd.to_datetime(f249)
        df249.index = df249['Fecha']
        df249['Casos'] = nc249
        n=df249
    if opcion == 250:
        for i in range(len(c250)):
            if i == 0:
                nc250.append(c250[i])
            else:
                nc250.append(c250[i]-c250[i-1])
        df250 = pd.DataFrame()
        df250['Fecha'] = pd.to_datetime(f250)
        df250.index = df250['Fecha']
        df250['Casos'] = nc250
        n=df250
    if opcion == 251:
        for i in range(len(c251)):
            if i == 0:
                nc251.append(c251[i])
            else:
                nc251.append(c251[i]-c251[i-1])
        df251 = pd.DataFrame()
        df251['Fecha'] = pd.to_datetime(f251)
        df251.index = df251['Fecha']
        df251['Casos'] = nc251
        n=df251
    if opcion == 252:
        for i in range(len(c252)):
            if i == 0:
                nc252.append(c252[i])
            else:
                nc252.append(c252[i]-c252[i-1])
        df252 = pd.DataFrame()
        df252['Fecha'] = pd.to_datetime(f252)
        df252.index = df252['Fecha']
        df252['Casos'] = nc252
        n=df252
    if opcion == 253:
        for i in range(len(c253)):
            if i == 0:
                nc253.append(c253[i])
            else:
                nc253.append(c253[i]-c253[i-1])
        df253 = pd.DataFrame()
        df253['Fecha'] = pd.to_datetime(f253)
        df253.index = df253['Fecha']
        df253['Casos'] = nc253
        n=df253
    if opcion == 254:
        for i in range(len(c254)):
            if i == 0:
                nc254.append(c254[i])
            else:
                nc254.append(c254[i]-c254[i-1])
        df254 = pd.DataFrame()
        df254['Fecha'] = pd.to_datetime(f254)
        df254.index = df254['Fecha']
        df254['Casos'] = nc254
        n=df254
    if opcion == 255:
        for i in range(len(c255)):
            if i == 0:
                nc255.append(c255[i])
            else:
                nc255.append(c255[i]-c255[i-1])
        df255 = pd.DataFrame()
        df255['Fecha'] = pd.to_datetime(f255)
        df255.index = df255['Fecha']
        df255['Casos'] = nc255
        n=df255
    if opcion == 256:
        for i in range(len(c256)):
            if i == 0:
                nc256.append(c256[i])
            else:
                nc256.append(c256[i]-c256[i-1])
        df256 = pd.DataFrame()
        df256['Fecha'] = pd.to_datetime(f256)
        df256.index = df256['Fecha']
        df256['Casos'] = nc256
        n=df256
    if opcion == 257:
        for i in range(len(c257)):
            if i == 0:
                nc257.append(c257[i])
            else:
                nc257.append(c257[i]-c257[i-1])
        df257 = pd.DataFrame()
        df257['Fecha'] = pd.to_datetime(f257)
        df257.index = df257['Fecha']
        df257['Casos'] = nc257
        n=df257
    if opcion == 258:
        for i in range(len(c258)):
            if i == 0:
                nc258.append(c258[i])
            else:
                nc258.append(c258[i]-c258[i-1])
        df258 = pd.DataFrame()
        df258['Fecha'] = pd.to_datetime(f258)
        df258.index = df258['Fecha']
        df258['Casos'] = nc258
        n=df258
    if opcion == 259:
        for i in range(len(c259)):
            if i == 0:
                nc259.append(c259[i])
            else:
                nc259.append(c259[i]-c259[i-1])
        df259 = pd.DataFrame()
        df259['Fecha'] = pd.to_datetime(f259)
        df259.index = df259['Fecha']
        df259['Casos'] = nc259
        n=df259
    if opcion == 260:
        for i in range(len(c260)):
            if i == 0:
                nc260.append(c260[i])
            else:
                nc260.append(c260[i]-c260[i-1])
        df260 = pd.DataFrame()
        df260['Fecha'] = pd.to_datetime(f260)
        df260.index = df260['Fecha']
        df260['Casos'] = nc260
        n=df260
    if opcion == 261:
        for i in range(len(c261)):
            if i == 0:
                nc261.append(c261[i])
            else:
                nc261.append(c261[i]-c261[i-1])
        df261 = pd.DataFrame()
        df261['Fecha'] = pd.to_datetime(f261)
        df261.index = df261['Fecha']
        df261['Casos'] = nc261
        n=df261
    if opcion == 262:
        for i in range(len(c262)):
            if i == 0:
                nc262.append(c262[i])
            else:
                nc262.append(c262[i]-c262[i-1])
        df262 = pd.DataFrame()
        df262['Fecha'] = pd.to_datetime(f262)
        df262.index = df262['Fecha']
        df262['Casos'] = nc262
        n=df262
    if opcion == 263:
        for i in range(len(c263)):
            if i == 0:
                nc263.append(c263[i])
            else:
                nc263.append(c263[i]-c263[i-1])
        df263 = pd.DataFrame()
        df263['Fecha'] = pd.to_datetime(f263)
        df263.index = df263['Fecha']
        df263['Casos'] = nc263
        n=df263
    if opcion == 264:
        for i in range(len(c264)):
            if i == 0:
                nc264.append(c264[i])
            else:
                nc264.append(c264[i]-c264[i-1])
        df264 = pd.DataFrame()
        df264['Fecha'] = pd.to_datetime(f264)
        df264.index = df264['Fecha']
        df264['Casos'] = nc264
        n=df264
    if opcion == 265:
        for i in range(len(c265)):
            if i == 0:
                nc265.append(c265[i])
            else:
                nc265.append(c265[i]-c265[i-1])
        df265 = pd.DataFrame()
        df265['Fecha'] = pd.to_datetime(f265)
        df265.index = df265['Fecha']
        df265['Casos'] = nc265
        n=df265
    if opcion == 266:
        for i in range(len(c266)):
            if i == 0:
                nc266.append(c266[i])
            else:
                nc266.append(c266[i]-c266[i-1])
        df266 = pd.DataFrame()
        df266['Fecha'] = pd.to_datetime(f266)
        df266.index = df266['Fecha']
        df266['Casos'] = nc266
        n=df266
    if opcion == 267:
        for i in range(len(c267)):
            if i == 0:
                nc267.append(c267[i])
            else:
                nc267.append(c267[i]-c267[i-1])
        df267 = pd.DataFrame()
        df267['Fecha'] = pd.to_datetime(f267)
        df267.index = df267['Fecha']
        df267['Casos'] = nc267
        n=df267
    if opcion == 268:
        for i in range(len(c268)):
            if i == 0:
                nc268.append(c268[i])
            else:
                nc268.append(c268[i]-c268[i-1])
        df268 = pd.DataFrame()
        df268['Fecha'] = pd.to_datetime(f268)
        df268.index = df268['Fecha']
        df268['Casos'] = nc268
        n=df268
    if opcion == 269:
        for i in range(len(c269)):
            if i == 0:
                nc269.append(c269[i])
            else:
                nc269.append(c269[i]-c269[i-1])
        df269 = pd.DataFrame()
        df269['Fecha'] = pd.to_datetime(f269)
        df269.index = df269['Fecha']
        df269['Casos'] = nc269
        n=df269
    if opcion == 270:
        for i in range(len(c270)):
            if i == 0:
                nc270.append(c270[i])
            else:
                nc270.append(c270[i]-c270[i-1])
        df270 = pd.DataFrame()
        df270['Fecha'] = pd.to_datetime(f270)
        df270.index = df270['Fecha']
        df270['Casos'] = nc270
        n=df270
    if opcion == 271:
        for i in range(len(c271)):
            if i == 0:
                nc271.append(c271[i])
            else:
                nc271.append(c271[i]-c271[i-1])
        df271 = pd.DataFrame()
        df271['Fecha'] = pd.to_datetime(f271)
        df271.index = df271['Fecha']
        df271['Casos'] = nc271
        n=df271
    if opcion == 272:
        for i in range(len(c272)):
            if i == 0:
                nc272.append(c272[i])
            else:
                nc272.append(c272[i]-c272[i-1])
        df272 = pd.DataFrame()
        df272['Fecha'] = pd.to_datetime(f272)
        df272.index = df272['Fecha']
        df272['Casos'] = nc272
        n=df272
    if opcion == 273:
        for i in range(len(c273)):
            if i == 0:
                nc273.append(c273[i])
            else:
                nc273.append(c273[i]-c273[i-1])
        df273 = pd.DataFrame()
        df273['Fecha'] = pd.to_datetime(f273)
        df273.index = df273['Fecha']
        df273['Casos'] = nc273
        n=df273
    if opcion == 274:
        for i in range(len(c274)):
            if i == 0:
                nc274.append(c274[i])
            else:
                nc274.append(c274[i]-c274[i-1])
        df274 = pd.DataFrame()
        df274['Fecha'] = pd.to_datetime(f274)
        df274.index = df274['Fecha']
        df274['Casos'] = nc274
        n=df274
    if opcion == 275:
        for i in range(len(c275)):
            if i == 0:
                nc275.append(c275[i])
            else:
                nc275.append(c275[i]-c275[i-1])
        df275 = pd.DataFrame()
        df275['Fecha'] = pd.to_datetime(f275)
        df275.index = df275['Fecha']
        df275['Casos'] = nc275
        n=df275
    if opcion == 276:
        for i in range(len(c276)):
            if i == 0:
                nc276.append(c276[i])
            else:
                nc276.append(c276[i]-c276[i-1])
        df276 = pd.DataFrame()
        df276['Fecha'] = pd.to_datetime(f276)
        df276.index = df276['Fecha']
        df276['Casos'] = nc276
        n=df276
    if opcion == 277:
        for i in range(len(c277)):
            if i == 0:
                nc277.append(c277[i])
            else:
                nc277.append(c277[i]-c277[i-1])
        df277 = pd.DataFrame()
        df277['Fecha'] = pd.to_datetime(f277)
        df277.index = df277['Fecha']
        df277['Casos'] = nc277
        n=df277
    if opcion == 278:
        for i in range(len(c278)):
            if i == 0:
                nc278.append(c278[i])
            else:
                nc278.append(c278[i]-c278[i-1])
        df278 = pd.DataFrame()
        df278['Fecha'] = pd.to_datetime(f278)
        df278.index = df278['Fecha']
        df278['Casos'] = nc278
        n=df278
    if opcion == 279:
        for i in range(len(c279)):
            if i == 0:
                nc279.append(c279[i])
            else:
                nc279.append(c279[i]-c279[i-1])
        df279 = pd.DataFrame()
        df279['Fecha'] = pd.to_datetime(f279)
        df279.index = df279['Fecha']
        df279['Casos'] = nc279
        n=df279
    if opcion == 280:
        for i in range(len(c280)):
            if i == 0:
                nc280.append(c280[i])
            else:
                nc280.append(c280[i]-c280[i-1])
        df280 = pd.DataFrame()
        df280['Fecha'] = pd.to_datetime(f280)
        df280.index = df280['Fecha']
        df280['Casos'] = nc280
        n=df280
    if opcion == 281:
        for i in range(len(c281)):
            if i == 0:
                nc281.append(c281[i])
            else:
                nc281.append(c281[i]-c281[i-1])
        df281 = pd.DataFrame()
        df281['Fecha'] = pd.to_datetime(f281)
        df281.index = df281['Fecha']
        df281['Casos'] = nc281
        n=df281
    if opcion == 282:
        for i in range(len(c282)):
            if i == 0:
                nc282.append(c282[i])
            else:
                nc282.append(c282[i]-c282[i-1])
        df282 = pd.DataFrame()
        df282['Fecha'] = pd.to_datetime(f282)
        df282.index = df282['Fecha']
        df282['Casos'] = nc282
        n=df282
    if opcion == 283:
        for i in range(len(c283)):
            if i == 0:
                nc283.append(c283[i])
            else:
                nc283.append(c283[i]-c283[i-1])
        df283 = pd.DataFrame()
        df283['Fecha'] = pd.to_datetime(f283)
        df283.index = df283['Fecha']
        df283['Casos'] = nc283
        n=df283
    if opcion == 284:
        for i in range(len(c284)):
            if i == 0:
                nc284.append(c284[i])
            else:
                nc284.append(c284[i]-c284[i-1])
        df284 = pd.DataFrame()
        df284['Fecha'] = pd.to_datetime(f284)
        df284.index = df284['Fecha']
        df284['Casos'] = nc284
        n=df284
    if opcion == 285:
        for i in range(len(c285)):
            if i == 0:
                nc285.append(c285[i])
            else:
                nc285.append(c285[i]-c285[i-1])
        df285 = pd.DataFrame()
        df285['Fecha'] = pd.to_datetime(f285)
        df285.index = df285['Fecha']
        df285['Casos'] = nc285
        n=df285
    if opcion == 286:
        for i in range(len(c286)):
            if i == 0:
                nc286.append(c286[i])
            else:
                nc286.append(c286[i]-c286[i-1])
        df286 = pd.DataFrame()
        df286['Fecha'] = pd.to_datetime(f286)
        df286.index = df286['Fecha']
        df286['Casos'] = nc286
        n=df286
    if opcion == 287:
        for i in range(len(c287)):
            if i == 0:
                nc287.append(c287[i])
            else:
                nc287.append(c287[i]-c287[i-1])
        df287 = pd.DataFrame()
        df287['Fecha'] = pd.to_datetime(f287)
        df287.index = df287['Fecha']
        df287['Casos'] = nc287
        n=df287
    if opcion == 288:
        for i in range(len(c288)):
            if i == 0:
                nc288.append(c288[i])
            else:
                nc288.append(c288[i]-c288[i-1])
        df288 = pd.DataFrame()
        df288['Fecha'] = pd.to_datetime(f288)
        df288.index = df288['Fecha']
        df288['Casos'] = nc288
        n=df288
    if opcion == 289:
        for i in range(len(c289)):
            if i == 0:
                nc289.append(c289[i])
            else:
                nc289.append(c289[i]-c289[i-1])
        df289 = pd.DataFrame()
        df289['Fecha'] = pd.to_datetime(f289)
        df289.index = df289['Fecha']
        df289['Casos'] = nc289
        n=df289
    if opcion == 290:
        for i in range(len(c290)):
            if i == 0:
                nc290.append(c290[i])
            else:
                nc290.append(c290[i]-c290[i-1])
        df290 = pd.DataFrame()
        df290['Fecha'] = pd.to_datetime(f290)
        df290.index = df290['Fecha']
        df290['Casos'] = nc290
        n=df290
    if opcion == 291:
        for i in range(len(c291)):
            if i == 0:
                nc291.append(c291[i])
            else:
                nc291.append(c291[i]-c291[i-1])
        df291 = pd.DataFrame()
        df291['Fecha'] = pd.to_datetime(f291)
        df291.index = df291['Fecha']
        df291['Casos'] = nc291
        n=df291
    if opcion == 292:
        for i in range(len(c292)):
            if i == 0:
                nc292.append(c292[i])
            else:
                nc292.append(c292[i]-c292[i-1])
        df292 = pd.DataFrame()
        df292['Fecha'] = pd.to_datetime(f292)
        df292.index = df292['Fecha']
        df292['Casos'] = nc292
        n=df292
    if opcion == 293:
        for i in range(len(c293)):
            if i == 0:
                nc293.append(c293[i])
            else:
                nc293.append(c293[i]-c293[i-1])
        df293 = pd.DataFrame()
        df293['Fecha'] = pd.to_datetime(f293)
        df293.index = df293['Fecha']
        df293['Casos'] = nc293
        n=df293
    if opcion == 294:
        for i in range(len(c294)):
            if i == 0:
                nc294.append(c294[i])
            else:
                nc294.append(c294[i]-c294[i-1])
        df294 = pd.DataFrame()
        df294['Fecha'] = pd.to_datetime(f294)
        df294.index = df294['Fecha']
        df294['Casos'] = nc294
        n=df294
    if opcion == 295:
        for i in range(len(c295)):
            if i == 0:
                nc295.append(c295[i])
            else:
                nc295.append(c295[i]-c295[i-1])
        df295 = pd.DataFrame()
        df295['Fecha'] = pd.to_datetime(f295)
        df295.index = df295['Fecha']
        df295['Casos'] = nc295
        n=df295
    if opcion == 296:
        for i in range(len(c296)):
            if i == 0:
                nc296.append(c296[i])
            else:
                nc296.append(c296[i]-c296[i-1])
        df296 = pd.DataFrame()
        df296['Fecha'] = pd.to_datetime(f296)
        df296.index = df296['Fecha']
        df296['Casos'] = nc296
        n=df296
    if opcion == 297:
        for i in range(len(c297)):
            if i == 0:
                nc297.append(c297[i])
            else:
                nc297.append(c297[i]-c297[i-1])
        df297 = pd.DataFrame()
        df297['Fecha'] = pd.to_datetime(f297)
        df297.index = df297['Fecha']
        df297['Casos'] = nc297
        n=df297
    if opcion == 298:
        for i in range(len(c298)):
            if i == 0:
                nc298.append(c298[i])
            else:
                nc298.append(c298[i]-c298[i-1])
        df298 = pd.DataFrame()
        df298['Fecha'] = pd.to_datetime(f298)
        df298.index = df298['Fecha']
        df298['Casos'] = nc298
        n=df298
    if opcion == 299:
        for i in range(len(c299)):
            if i == 0:
                nc299.append(c299[i])
            else:
                nc299.append(c299[i]-c299[i-1])
        df299 = pd.DataFrame()
        df299['Fecha'] = pd.to_datetime(f299)
        df299.index = df299['Fecha']
        df299['Casos'] = nc299
        n=df299
    if opcion == 300:
        for i in range(len(c300)):
            if i == 0:
                nc300.append(c300[i])
            else:
                nc300.append(c300[i]-c300[i-1])
        df300 = pd.DataFrame()
        df300['Fecha'] = pd.to_datetime(f300)
        df300.index = df300['Fecha']
        df300['Casos'] = nc300
        n=df300
    if opcion == 301:
        for i in range(len(c301)):
            if i == 0:
                nc301.append(c301[i])
            else:
                nc301.append(c301[i]-c301[i-1])
        df301 = pd.DataFrame()
        df301['Fecha'] = pd.to_datetime(f301)
        df301.index = df301['Fecha']
        df301['Casos'] = nc301
        n=df301
    if opcion == 302:
        for i in range(len(c302)):
            if i == 0:
                nc302.append(c302[i])
            else:
                nc302.append(c302[i]-c302[i-1])
        df302 = pd.DataFrame()
        df302['Fecha'] = pd.to_datetime(f302)
        df302.index = df302['Fecha']
        df302['Casos'] = nc302
        n=df302
    if opcion == 303:
        for i in range(len(c303)):
            if i == 0:
                nc303.append(c303[i])
            else:
                nc303.append(c303[i]-c303[i-1])
        df303 = pd.DataFrame()
        df303['Fecha'] = pd.to_datetime(f303)
        df303.index = df303['Fecha']
        df303['Casos'] = nc303
        n=df303
    if opcion == 304:
        for i in range(len(c304)):
            if i == 0:
                nc304.append(c304[i])
            else:
                nc304.append(c304[i]-c304[i-1])
        df304 = pd.DataFrame()
        df304['Fecha'] = pd.to_datetime(f304)
        df304.index = df304['Fecha']
        df304['Casos'] = nc304
        n=df304
    if opcion == 305:
        for i in range(len(c305)):
            if i == 0:
                nc305.append(c305[i])
            else:
                nc305.append(c305[i]-c305[i-1])
        df305 = pd.DataFrame()
        df305['Fecha'] = pd.to_datetime(f305)
        df305.index = df305['Fecha']
        df305['Casos'] = nc305
        n=df305
    if opcion == 306:
        for i in range(len(c306)):
            if i == 0:
                nc306.append(c306[i])
            else:
                nc306.append(c306[i]-c306[i-1])
        df306 = pd.DataFrame()
        df306['Fecha'] = pd.to_datetime(f306)
        df306.index = df306['Fecha']
        df306['Casos'] = nc306
        n=df306
    if opcion == 307:
        for i in range(len(c307)):
            if i == 0:
                nc307.append(c307[i])
            else:
                nc307.append(c307[i]-c307[i-1])
        df307 = pd.DataFrame()
        df307['Fecha'] = pd.to_datetime(f307)
        df307.index = df307['Fecha']
        df307['Casos'] = nc307
        n=df307
    if opcion == 308:
        for i in range(len(c308)):
            if i == 0:
                nc308.append(c308[i])
            else:
                nc308.append(c308[i]-c308[i-1])
        df308 = pd.DataFrame()
        df308['Fecha'] = pd.to_datetime(f308)
        df308.index = df308['Fecha']
        df308['Casos'] = nc308
        n=df308
    if opcion == 309:
        for i in range(len(c309)):
            if i == 0:
                nc309.append(c309[i])
            else:
                nc309.append(c309[i]-c309[i-1])
        df309 = pd.DataFrame()
        df309['Fecha'] = pd.to_datetime(f309)
        df309.index = df309['Fecha']
        df309['Casos'] = nc309
        n=df309
    if opcion == 310:
        for i in range(len(c310)):
            if i == 0:
                nc310.append(c310[i])
            else:
                nc310.append(c310[i]-c310[i-1])
        df310 = pd.DataFrame()
        df310['Fecha'] = pd.to_datetime(f310)
        df310.index = df310['Fecha']
        df310['Casos'] = nc310
        n=df310
    if opcion == 311:
        for i in range(len(c311)):
            if i == 0:
                nc311.append(c311[i])
            else:
                nc311.append(c311[i]-c311[i-1])
        df311 = pd.DataFrame()
        df311['Fecha'] = pd.to_datetime(f311)
        df311.index = df311['Fecha']
        df311['Casos'] = nc311
        n=df311
    if opcion == 312:
        for i in range(len(c312)):
            if i == 0:
                nc312.append(c312[i])
            else:
                nc312.append(c312[i]-c312[i-1])
        df312 = pd.DataFrame()
        df312['Fecha'] = pd.to_datetime(f312)
        df312.index = df312['Fecha']
        df312['Casos'] = nc312
        n=df312
    if opcion == 313:
        for i in range(len(c313)):
            if i == 0:
                nc313.append(c313[i])
            else:
                nc313.append(c313[i]-c313[i-1])
        df313 = pd.DataFrame()
        df313['Fecha'] = pd.to_datetime(f313)
        df313.index = df313['Fecha']
        df313['Casos'] = nc313
        n=df313
    if opcion == 314:
        for i in range(len(c314)):
            if i == 0:
                nc314.append(c314[i])
            else:
                nc314.append(c314[i]-c314[i-1])
        df314 = pd.DataFrame()
        df314['Fecha'] = pd.to_datetime(f314)
        df314.index = df314['Fecha']
        df314['Casos'] = nc314
        n=df314
    if opcion == 315:
        for i in range(len(c315)):
            if i == 0:
                nc315.append(c315[i])
            else:
                nc315.append(c315[i]-c315[i-1])
        df315 = pd.DataFrame()
        df315['Fecha'] = pd.to_datetime(f315)
        df315.index = df315['Fecha']
        df315['Casos'] = nc315
        n=df315
    if opcion == 316:
        for i in range(len(c316)):
            if i == 0:
                nc316.append(c316[i])
            else:
                nc316.append(c316[i]-c316[i-1])
        df316 = pd.DataFrame()
        df316['Fecha'] = pd.to_datetime(f316)
        df316.index = df316['Fecha']
        df316['Casos'] = nc316
        n=df316
    if opcion == 317:
        for i in range(len(c317)):
            if i == 0:
                nc317.append(c317[i])
            else:
                nc317.append(c317[i]-c317[i-1])
        df317 = pd.DataFrame()
        df317['Fecha'] = pd.to_datetime(f317)
        df317.index = df317['Fecha']
        df317['Casos'] = nc317
        n=df317
    if opcion == 318:
        for i in range(len(c318)):
            if i == 0:
                nc318.append(c318[i])
            else:
                nc318.append(c318[i]-c318[i-1])
        df318 = pd.DataFrame()
        df318['Fecha'] = pd.to_datetime(f318)
        df318.index = df318['Fecha']
        df318['Casos'] = nc318
        n=df318
    if opcion == 319:
        for i in range(len(c319)):
            if i == 0:
                nc319.append(c319[i])
            else:
                nc319.append(c319[i]-c319[i-1])
        df319 = pd.DataFrame()
        df319['Fecha'] = pd.to_datetime(f319)
        df319.index = df319['Fecha']
        df319['Casos'] = nc319
        n=df319
    if opcion == 320:
        for i in range(len(c320)):
            if i == 0:
                nc320.append(c320[i])
            else:
                nc320.append(c320[i]-c320[i-1])
        df320 = pd.DataFrame()
        df320['Fecha'] = pd.to_datetime(f320)
        df320.index = df320['Fecha']
        df320['Casos'] = nc320
        n=df320
    if opcion == 321:
        for i in range(len(c321)):
            if i == 0:
                nc321.append(c321[i])
            else:
                nc321.append(c321[i]-c321[i-1])
        df321 = pd.DataFrame()
        df321['Fecha'] = pd.to_datetime(f321)
        df321.index = df321['Fecha']
        df321['Casos'] = nc321
        n=df321
    if opcion == 322:
        for i in range(len(c322)):
            if i == 0:
                nc322.append(c322[i])
            else:
                nc322.append(c322[i]-c322[i-1])
        df322 = pd.DataFrame()
        df322['Fecha'] = pd.to_datetime(f322)
        df322.index = df322['Fecha']
        df322['Casos'] = nc322
        n=df322
    if opcion == 323:
        for i in range(len(c323)):
            if i == 0:
                nc323.append(c323[i])
            else:
                nc323.append(c323[i]-c323[i-1])
        df323 = pd.DataFrame()
        df323['Fecha'] = pd.to_datetime(f323)
        df323.index = df323['Fecha']
        df323['Casos'] = nc323
        n=df323
    if opcion == 324:
        for i in range(len(c324)):
            if i == 0:
                nc324.append(c324[i])
            else:
                nc324.append(c324[i]-c324[i-1])
        df324 = pd.DataFrame()
        df324['Fecha'] = pd.to_datetime(f324)
        df324.index = df324['Fecha']
        df324['Casos'] = nc324
        n=df324
    if opcion == 325:
        for i in range(len(c325)):
            if i == 0:
                nc325.append(c325[i])
            else:
                nc325.append(c325[i]-c325[i-1])
        df325 = pd.DataFrame()
        df325['Fecha'] = pd.to_datetime(f325)
        df325.index = df325['Fecha']
        df325['Casos'] = nc325
        n=df325
    if opcion == 326:
        for i in range(len(c326)):
            if i == 0:
                nc326.append(c326[i])
            else:
                nc326.append(c326[i]-c326[i-1])
        df326 = pd.DataFrame()
        df326['Fecha'] = pd.to_datetime(f326)
        df326.index = df326['Fecha']
        df326['Casos'] = nc326
        n=df326
    if opcion == 327:
        for i in range(len(c327)):
            if i == 0:
                nc327.append(c327[i])
            else:
                nc327.append(c327[i]-c327[i-1])
        df327 = pd.DataFrame()
        df327['Fecha'] = pd.to_datetime(f327)
        df327.index = df327['Fecha']
        df327['Casos'] = nc327
        n=df327
    if opcion == 328:
        for i in range(len(c328)):
            if i == 0:
                nc328.append(c328[i])
            else:
                nc328.append(c328[i]-c328[i-1])
        df328 = pd.DataFrame()
        df328['Fecha'] = pd.to_datetime(f328)
        df328.index = df328['Fecha']
        df328['Casos'] = nc328
        n=df328
    if opcion == 329:
        for i in range(len(c329)):
            if i == 0:
                nc329.append(c329[i])
            else:
                nc329.append(c329[i]-c329[i-1])
        df329 = pd.DataFrame()
        df329['Fecha'] = pd.to_datetime(f329)
        df329.index = df329['Fecha']
        df329['Casos'] = nc329
        n=df329
    if opcion == 330:
        for i in range(len(c330)):
            if i == 0:
                nc330.append(c330[i])
            else:
                nc330.append(c330[i]-c330[i-1])
        df330 = pd.DataFrame()
        df330['Fecha'] = pd.to_datetime(f330)
        df330.index = df330['Fecha']
        df330['Casos'] = nc330
        n=df330
    if opcion == 331:
        for i in range(len(c331)):
            if i == 0:
                nc331.append(c331[i])
            else:
                nc331.append(c331[i]-c331[i-1])
        df331 = pd.DataFrame()
        df331['Fecha'] = pd.to_datetime(f331)
        df331.index = df331['Fecha']
        df331['Casos'] = nc331
        n=df331
    if opcion == 332:
        for i in range(len(c332)):
            if i == 0:
                nc332.append(c332[i])
            else:
                nc332.append(c332[i]-c332[i-1])
        df332 = pd.DataFrame()
        df332['Fecha'] = pd.to_datetime(f332)
        df332.index = df332['Fecha']
        df332['Casos'] = nc332
        n=df332
    if opcion == 333:
        for i in range(len(c333)):
            if i == 0:
                nc333.append(c333[i])
            else:
                nc333.append(c333[i]-c333[i-1])
        df333 = pd.DataFrame()
        df333['Fecha'] = pd.to_datetime(f333)
        df333.index = df333['Fecha']
        df333['Casos'] = nc333
        n=df333
    if opcion == 334:
        for i in range(len(c334)):
            if i == 0:
                nc334.append(c334[i])
            else:
                nc334.append(c334[i]-c334[i-1])
        df334 = pd.DataFrame()
        df334['Fecha'] = pd.to_datetime(f334)
        df334.index = df334['Fecha']
        df334['Casos'] = nc334
        n=df334
    if opcion == 335:
        for i in range(len(c335)):
            if i == 0:
                nc335.append(c335[i])
            else:
                nc335.append(c335[i]-c335[i-1])
        df335 = pd.DataFrame()
        df335['Fecha'] = pd.to_datetime(f335)
        df335.index = df335['Fecha']
        df335['Casos'] = nc335
        n=df335
    if opcion == 336:
        for i in range(len(c336)):
            if i == 0:
                nc336.append(c336[i])
            else:
                nc336.append(c336[i]-c336[i-1])
        df336 = pd.DataFrame()
        df336['Fecha'] = pd.to_datetime(f336)
        df336.index = df336['Fecha']
        df336['Casos'] = nc336
        n=df336
    if opcion == 337:
        for i in range(len(c337)):
            if i == 0:
                nc337.append(c337[i])
            else:
                nc337.append(c337[i]-c337[i-1])
        df337 = pd.DataFrame()
        df337['Fecha'] = pd.to_datetime(f337)
        df337.index = df337['Fecha']
        df337['Casos'] = nc337
        n=df337
    if opcion == 338:
        for i in range(len(c338)):
            if i == 0:
                nc338.append(c338[i])
            else:
                nc338.append(c338[i]-c338[i-1])
        df338 = pd.DataFrame()
        df338['Fecha'] = pd.to_datetime(f338)
        df338.index = df338['Fecha']
        df338['Casos'] = nc338
        n=df338
    if opcion == 339:
        for i in range(len(c339)):
            if i == 0:
                nc339.append(c339[i])
            else:
                nc339.append(c339[i]-c339[i-1])
        df339 = pd.DataFrame()
        df339['Fecha'] = pd.to_datetime(f339)
        df339.index = df339['Fecha']
        df339['Casos'] = nc339
        n=df339
    if opcion == 340:
        for i in range(len(c340)):
            if i == 0:
                nc340.append(c340[i])
            else:
                nc340.append(c340[i]-c340[i-1])
        df340 = pd.DataFrame()
        df340['Fecha'] = pd.to_datetime(f340)
        df340.index = df340['Fecha']
        df340['Casos'] = nc340
        n=df340
    if opcion == 341:
        for i in range(len(c341)):
            if i == 0:
                nc341.append(c341[i])
            else:
                nc341.append(c341[i]-c341[i-1])
        df341 = pd.DataFrame()
        df341['Fecha'] = pd.to_datetime(f341)
        df341.index = df341['Fecha']
        df341['Casos'] = nc341
        n=df341
    if opcion == 342:
        for i in range(len(c342)):
            if i == 0:
                nc342.append(c342[i])
            else:
                nc342.append(c342[i]-c342[i-1])
        df342 = pd.DataFrame()
        df342['Fecha'] = pd.to_datetime(f342)
        df342.index = df342['Fecha']
        df342['Casos'] = nc342
        n=df342
    if opcion == 343:
        for i in range(len(c343)):
            if i == 0:
                nc343.append(c343[i])
            else:
                nc343.append(c343[i]-c343[i-1])
        df343 = pd.DataFrame()
        df343['Fecha'] = pd.to_datetime(f343)
        df343.index = df343['Fecha']
        df343['Casos'] = nc343
        n=df343
    if opcion == 344:
        for i in range(len(c344)):
            if i == 0:
                nc344.append(c344[i])
            else:
                nc344.append(c344[i]-c344[i-1])
        df344 = pd.DataFrame()
        df344['Fecha'] = pd.to_datetime(f344)
        df344.index = df344['Fecha']
        df344['Casos'] = nc344
        n=df344
    if opcion == 345:
        for i in range(len(c345)):
            if i == 0:
                nc345.append(c345[i])
            else:
                nc345.append(c345[i]-c345[i-1])
        df345 = pd.DataFrame()
        df345['Fecha'] = pd.to_datetime(f345)
        df345.index = df345['Fecha']
        df345['Casos'] = nc345
        n=df345
    if opcion == 346:
        for i in range(len(c346)):
            if i == 0:
                nc346.append(c346[i])
            else:
                nc346.append(c346[i]-c346[i-1])
        df346 = pd.DataFrame()
        df346['Fecha'] = pd.to_datetime(f346)
        df346.index = df346['Fecha']
        df346['Casos'] = nc346
        n=df346
    if opcion == 347:
        for i in range(len(c347)):
            if i == 0:
                nc347.append(c347[i])
            else:
                nc347.append(c347[i]-c347[i-1])
        df347 = pd.DataFrame()
        df347['Fecha'] = pd.to_datetime(f347)
        df347.index = df347['Fecha']
        df347['Casos'] = nc347
        n=df347
    if opcion == 348:
        for i in range(len(c348)):
            if i == 0:
                nc348.append(c348[i])
            else:
                nc348.append(c348[i]-c348[i-1])
        df348 = pd.DataFrame()
        df348['Fecha'] = pd.to_datetime(f348)
        df348.index = df348['Fecha']
        df348['Casos'] = nc348
        n=df348
    if opcion == 349:
        for i in range(len(c349)):
            if i == 0:
                nc349.append(c349[i])
            else:
                nc349.append(c349[i]-c349[i-1])
        df349 = pd.DataFrame()
        df349['Fecha'] = pd.to_datetime(f349)
        df349.index = df349['Fecha']
        df349['Casos'] = nc349
        n=df349
    if opcion == 350:
        for i in range(len(c350)):
            if i == 0:
                nc350.append(c350[i])
            else:
                nc350.append(c350[i]-c350[i-1])
        df350 = pd.DataFrame()
        df350['Fecha'] = pd.to_datetime(f350)
        df350.index = df350['Fecha']
        df350['Casos'] = nc350
        n=df350
    if opcion == 351:
        for i in range(len(c351)):
            if i == 0:
                nc351.append(c351[i])
            else:
                nc351.append(c351[i]-c351[i-1])
        df351 = pd.DataFrame()
        df351['Fecha'] = pd.to_datetime(f351)
        df351.index = df351['Fecha']
        df351['Casos'] = nc351
        n=df351
    if opcion == 352:
        for i in range(len(c352)):
            if i == 0:
                nc352.append(c352[i])
            else:
                nc352.append(c352[i]-c352[i-1])
        df352 = pd.DataFrame()
        df352['Fecha'] = pd.to_datetime(f352)
        df352.index = df352['Fecha']
        df352['Casos'] = nc352
        n=df352
    if opcion == 353:
        for i in range(len(c353)):
            if i == 0:
                nc353.append(c353[i])
            else:
                nc353.append(c353[i]-c353[i-1])
        df353 = pd.DataFrame()
        df353['Fecha'] = pd.to_datetime(f353)
        df353.index = df353['Fecha']
        df353['Casos'] = nc353
        n=df353
    if opcion == 354:
        for i in range(len(c354)):
            if i == 0:
                nc354.append(c354[i])
            else:
                nc354.append(c354[i]-c354[i-1])
        df354 = pd.DataFrame()
        df354['Fecha'] = pd.to_datetime(f354)
        df354.index = df354['Fecha']
        df354['Casos'] = nc354
        n=df354
    if opcion == 355:
        for i in range(len(c355)):
            if i == 0:
                nc355.append(c355[i])
            else:
                nc355.append(c355[i]-c355[i-1])
        df355 = pd.DataFrame()
        df355['Fecha'] = pd.to_datetime(f355)
        df355.index = df355['Fecha']
        df355['Casos'] = nc355
        n=df355
    if opcion == 356:
        for i in range(len(c356)):
            if i == 0:
                nc356.append(c356[i])
            else:
                nc356.append(c356[i]-c356[i-1])
        df356 = pd.DataFrame()
        df356['Fecha'] = pd.to_datetime(f356)
        df356.index = df356['Fecha']
        df356['Casos'] = nc356
        n=df356
    if opcion == 357:
        for i in range(len(c357)):
            if i == 0:
                nc357.append(c357[i])
            else:
                nc357.append(c357[i]-c357[i-1])
        df357 = pd.DataFrame()
        df357['Fecha'] = pd.to_datetime(f357)
        df357.index = df357['Fecha']
        df357['Casos'] = nc357
        n=df357
    if opcion == 358:
        for i in range(len(c358)):
            if i == 0:
                nc358.append(c358[i])
            else:
                nc358.append(c358[i]-c358[i-1])
        df358 = pd.DataFrame()
        df358['Fecha'] = pd.to_datetime(f358)
        df358.index = df358['Fecha']
        df358['Casos'] = nc358
        n=df358
    if opcion == 359:
        for i in range(len(c359)):
            if i == 0:
                nc359.append(c359[i])
            else:
                nc359.append(c359[i]-c359[i-1])
        df359 = pd.DataFrame()
        df359['Fecha'] = pd.to_datetime(f359)
        df359.index = df359['Fecha']
        df359['Casos'] = nc359
        n=df359
    if opcion == 360:
        for i in range(len(c360)):
            if i == 0:
                nc360.append(c360[i])
            else:
                nc360.append(c360[i]-c360[i-1])
        df360 = pd.DataFrame()
        df360['Fecha'] = pd.to_datetime(f360)
        df360.index = df360['Fecha']
        df360['Casos'] = nc360
        n=df360
    if opcion == 361:
        for i in range(len(c361)):
            if i == 0:
                nc361.append(c361[i])
            else:
                nc361.append(c361[i]-c361[i-1])
        df361 = pd.DataFrame()
        df361['Fecha'] = pd.to_datetime(f361)
        df361.index = df361['Fecha']
        df361['Casos'] = nc361
        n=df361


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

    for i in range(len(datos_invertidos)):
        datos_invertidos[i] = int(datos_invertidos[i])


    prediccion = pd.DataFrame(datos_invertidos)
    prediccion.columns = ['pronostico']
    #prediccion.plot()
    #plt.show()
    return prediccion