#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymongo
import os.path
import json


conexion = pymongo.MongoClient("localhost", 27017)

db = conexion.PrediccionCovid
fallecidos = db.Comuna_Fallecidos
dato ={}
region = ""
comuna = ""
x=""
y=""
n=0
if os.path.exists("json/CasosFallecidosPorComuna.json") == True:
    with open("json/CasosFallecidosPorComuna.json",encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():
            x=value
            region= key 
            for key, value in x.items():
                if( key!="Total"):
                    y= value
                    comuna = key
                    for key, value in y.items():
                        if(value!=""):
                            dato={'Region': region,
                                'Comuna': comuna,
                                'Fecha' : key,
                                'Fallecidos' : value 
                            }
                            with open ("json/json_fallecidos/"+region+"_"+comuna+"_"+key+".json", "w", encoding='utf8') as jsonFile:
                               jsonFile.write(json.dumps(dato, indent=4, ensure_ascii=False))

                            with open ("json/json_fallecidos/"+region+"_"+comuna+"_"+key+".json", "r", encoding="utf8") as jsonFile:
                                dat = json.load(jsonFile)
                                n = fallecidos.find({"Region": region, "Comuna": comuna, "Fecha": key}).count()
                                if (n==0):
                                    fallecidos.insert(dat)   


