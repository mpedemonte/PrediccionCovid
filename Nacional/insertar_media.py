import pymongo
import os.path
import json
conexion = pymongo.MongoClient("localhost", 27017)

db = conexion.PrediccionCovid
media_movil = db.Nacional_Media_Movil
dato ={}

if os.path.exists("MediaMovil.json") == True:
    with open("MediaMovil.json") as file:
        data = json.load(file)
        for key, value in data.items():
            if (key != "Region" and value != ""):
                dato = {'Fecha': key,
                        'Media': value}
                with open ("mediamovil/"+key+".json", "w") as jsonFile:
                    jsonFile.write(json.dumps(dato, indent=4))

                with open ("mediamovil/"+key+".json") as jsonFile:
                    dat = json.load(jsonFile)
                    x = media_movil.find({"Fecha": key}).count()
                    if (x==0):
                        media_movil.insert(dat)   



