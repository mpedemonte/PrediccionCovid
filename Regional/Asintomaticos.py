#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv, json

csvFilePath='CasosNuevosSinSintomas.csv'
jsonFilePath='Asintomaticos.json'

data={}
with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        id = rows['Region']
        data[id]= rows
del data['Total']

with open(jsonFilePath, 'w') as jsonFile:
    jsonFile.write(json.dumps(data,indent=4))

