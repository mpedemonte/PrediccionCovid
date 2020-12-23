#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv,json

aFecha=open('./CasosFallecidosPorComuna.csv','r', encoding='utf-8')
aReader=csv.reader(aFecha)

for row in aReader:
    largo=len(row)
    Fecha=row
    break

dDic = {}
data=open('./CasosFallecidosPorComuna.csv','r', encoding='utf-8')
reader=csv.reader(data)

aux=''
for row in reader:
    if aux!=row[0]:
        aux=row[0]
        dd={}
        dd[row[2]]={Fecha[i]:row[i] for i in range(5,largo)}
        
    else:
        dd[row[2]]={Fecha[i]:row[i] for i in range(5,largo)}  
    dDic[aux]=dd    
del dDic["Region"]
with open('./json/CasosFallecidosPorComuna.json', 'w', encoding="utf8") as file:
    json.dump(dDic,file, indent=4, ensure_ascii=False)