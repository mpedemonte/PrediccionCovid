import webbrowser


ListaUrl=["https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto47/MediaMovil.csv","https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto10/FallecidosEtario.csv","https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto26/CasosNuevosConSintomas.csv","https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto27/CasosNuevosSinSintomas.csv","https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto14/FallecidosCumulativo.csv","https://github.com/MinCiencia/Datos-COVID19/tree/master/output/producto2/","https://github.com/MinCiencia/Datos-COVID19/blob/master/output/producto38/CasosFallecidosPorComuna.csv"]
ListaNom=["MediaMovil","FallecidosEtario","CasosNuevosConSintomas","CasosNuevosSinSintomas","FallecidosCumulativo","Casos Confirmados","CasosFallecidosPorComuna"]
for i in range(len(ListaUrl)):
    url="https://minhaskamal.github.io/DownGit/#/home?url="+ListaUrl[i]+"&fileName="+ListaNom[i]
    webbrowser.open(url, new=2, autoraise=False)