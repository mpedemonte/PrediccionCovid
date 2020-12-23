import csv
import json

csvFilePath = "MediaMovil.csv"
jsonFilePath = "MediaMovil.json"

data = {}
x=0
data=""
with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        if x==16:
            print (rows)
            data = rows
        x = x+1

with open (jsonFilePath, "w") as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))

