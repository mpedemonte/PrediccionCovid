#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymongo
import os.path
import json
from os import listdir

def archivos(ruta = 'Casos Confirmados/json/'):
    return listdir(ruta)

conexion = pymongo.MongoClient("localhost", 27017)

db = conexion.PrediccionCovid
casos = db.Comuna_Casos
dato ={}
region = ""
comuna = ""
fecha = ""
x=""
y=""
z=""
n=0
archivo =(archivos())
print(archivo)
for i in archivo:
    with open("Casos Confirmados/json/"+i,encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():
            x=value
            fecha= key 
            for key, value in x.items():
                y=value
                region = key
                for key, value in y.items():
                    comuna = key
                    z=value
                    for key, value in z.items():
                        if(value!=""):
                            dato={'Region': region,
                                'Comuna': comuna,
                                'Fecha' : fecha,
                                'Casos' : value 
                            }
                            #with open ("Casos Confirmados/json/casos/"+region+"_"+comuna+"_"+fecha+".json", "w", encoding='utf8') as jsonFile:
                               #jsonFile.write(json.dumps(dato, indent=4, ensure_ascii=False))

                            #with open ("Casos Confirmados/json/casos/"+region+"_"+comuna+"_"+fecha+".json", "r", encoding="utf8") as jsonFile:
                                #dat = json.load(jsonFile)
                                #n = casos.find({"Region": region, "Comuna": comuna, "Fecha": fecha}).count()
                                #if (n==0):
                                    #casos.insert(dat)   

