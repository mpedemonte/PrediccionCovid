import csv,json
from os import listdir

def archivos(ruta = 'Casos Confirmados/'):
    return listdir(ruta)

nombres =(archivos())


for j in range(len(nombres)):
    dDic = {"2020-03-30":{}}
    data=open('./Casos Confirmados/'+nombres[j])
    reader=csv.reader(data)

    aux=''
    for row in reader:
        if aux!=row[0]:
            aux=row[0]
            dd={}
            dd[row[2]]={"Casos Confirmados":row[5]}
        else:
            dd[row[2]]={"Casos Confirmados":row[5]}
        dDic["2020-03-30"][aux]=dd
    del dDic["2020-03-30"]["Region"]
    with open('./Casos Confirmados/json/'+nombres[j][:-3]+'json','w') as file:
        json.dump(dDic,file, indent=4)


        