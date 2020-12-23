from os import listdir


##################################33
#Agrega un 2 a archivos bugeados con el nombre (020 en vez de 2020)
################################33333


def archivos(ruta = './'):
    return listdir(ruta)

nombres =(archivos())
nombres = nombres[:-2]

for j in range(len(nombres)):
    data=open('./'+nombres[j],'r', encoding='utf-8')
    reader=data.read()
    with open('./2'+nombres[j], 'w', encoding="utf8") as file:
        file.write(reader)