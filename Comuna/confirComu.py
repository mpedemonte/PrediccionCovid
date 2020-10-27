import csv,json
from os import listdir

def archivos(ruta = 'Casos Confirmados/'):
    return listdir(ruta)

nombres =(archivos())
nombres = nombres[:-1]

for j in range(len(nombres)):
    dDic = {nombres[j][:-21]:{}}
    #print (nombres[j])
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
        dDic[nombres[j][:-21]][aux]=dd
    del dDic[nombres[j][:-21]]["Region"]
    with open('./Casos Confirmados/json/'+nombres[j][:-3]+'json','w') as file:
        json.dump(dDic,file, indent=4)
       