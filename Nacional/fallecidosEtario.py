import csv
import json

csvFilePath = "FallecidosEtario.csv"

c=0
data=""
with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        data = rows
        print (rows["Grupo de edad"])
        with open ("fallecidos/"+str(c)+".json", "w") as jsonFile:
           jsonFile.write(json.dumps(data, indent=4))
        c=c+1


