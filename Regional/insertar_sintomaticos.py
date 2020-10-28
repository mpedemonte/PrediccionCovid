#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymongo
import os.path
import json


conexion = pymongo.MongoClient("localhost", 27017)

db = conexion.PrediccionCovid
sintomaticos = db.Region_Casos_Sintomaticos
dato ={}
region = ""
x=""
n=0
if os.path.exists("json/Sintomaticos.json") == True:
    with open("json/Sintomaticos.json",encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():
            x=value
            region= key 
            for key, value in x.items():
                if (key != "Region"):
                    dato={'Region': region,
                          'Fecha' : key,
                          'Casos' : value 
                          }
                    with open ("json/json_sintomaticos/"+region+"_"+key+".json", "w", encoding='utf8') as jsonFile:
                        jsonFile.write(json.dumps(dato, indent=4, ensure_ascii=False))

                    with open ("json/json_sintomaticos/"+region+"_"+key+".json", "r", encoding="utf8") as jsonFile:
                        dat = json.load(jsonFile)
                        n = sintomaticos.find({"Region": region, "Fecha": key}).count()
                        if (n==0):
                            sintomaticos.insert(dat)   


