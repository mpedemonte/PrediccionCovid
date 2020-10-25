import pymongo
import os.path
import json
from os import listdir

def archivos(ruta = 'fallecidos/'):
    return listdir(ruta)
conexion = pymongo.MongoClient("localhost", 27017)

db = conexion.PrediccionCovid
nacional_fallecidos = db.Nacional_Fallecidos
dato ={}
grupo=""
grup = 1
archivo =(archivos())
for i in archivo:
    with open("fallecidos/"+i) as file:
        data = json.load(file)
        for key, value in data.items():
            if (key == "Grupo de edad"):
                grupo = value
            else:
                dato = {'Grupo_Etario': grupo,
                        'Fecha': key,
                        'Media': value}
                print (dato)        
                with open ("fallecidos_json/"+str(grup)+"_"+key+".json", "w") as jsonFile:
                    jsonFile.write(json.dumps(dato, indent=4))

                with open ("fallecidos_json/"+str(grup)+"_"+key+".json") as jsonFile:
                    dat = json.load(jsonFile)
                    x = nacional_fallecidos.find({"Grupo_Etario": grupo, "Fecha": key}).count()
                    if (x==0):
                        nacional_fallecidos.insert(dat)  
    grup=grup+1        