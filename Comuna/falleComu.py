import csv,json

aFecha=open('./CasosFallecidosPorComuna.csv')
aReader=csv.reader(aFecha)

for row in aReader:
    Fecha=row
    break

dDic = {}
data=open('./CasosFallecidosPorComuna.csv')
reader=csv.reader(data)

aux=''
for row in reader:
    if aux!=row[0]:
        aux=row[0]
        dd={}
    else:
        dd[row[2]]={Fecha[i]:row[i] for i in range(5,38)}
    dDic[aux]=dd

with open('./json/CasosFallecidosPorComuna.json','w') as file:
    json.dump(dDic,file, indent=4)