#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv, json

csvFilePath='./csv/CasosNuevosConSintomas.csv'
jsonFilePath='./json/Sintomaticos.json'

data={}
with open(csvFilePath,'r', encoding='utf-8') as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        id = rows['Region']
        data[id]= rows
del data['Total']


with open(jsonFilePath, 'w', encoding="utf8") as jsonFile:
    jsonFile.write(json.dumps(data,indent=4, ensure_ascii=False))

