#CREACION ENTORNO VIRTUAL EN CONDA LLAMADO STREAMLIT
#       conda create -n streamlit -y


#ACTIVAR ENTORNO VIRTUAL
#          conda activate streamlit


#INSTALAR STREAMLIT
#    pip install streamlit



import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

#################### TITULO ###################
st.title('Predicción del covid-19 en Chile')
###############################################


#########################################################
#########################################################
#########################################################
####################  SIDEBAR  ##########################
#########################################################
#########################################################
#########################################################

st.sidebar.header("Navegación")

opcion_nav = st.sidebar.radio("",('Comuna', 'Región','Nacional'))



dDic_regiones={
    "Arica y Parinacota":1,
    "Tarapacá":2,
    "Antofagasta":3,
    "Atacama":4,
    "Coquimbo":5,
    "Valparaíso":6,
    "Metropolitana":7,
    "O’Higgins":8,
    "Maule":9,
    "Ñuble":10,
    "Biobío":11,
    "Araucanía":12,
    "Los Ríos":13,
    "Los Lagos":14,
    "Aysén":15,
    "Magallanes":16
}
aRegiones=["Arica y Parinacota",
    "Tarapacá",
    "Antofagasta",
    "Atacama",
    "Coquimbo",
    "Valparaíso",
    "Metropolitana",
    "O’Higgins",
    "Maule",
    "Ñuble",
    "Biobío",
    "Araucanía",
    "Los Ríos",
    "Los Lagos",
    "Aysén",
    "Magallanes"]

dDic_rango={
    "<=39":1,
    "40-49":2,
    "50-59":3,
    "60-69":4,
    "70-79":5,
    "80-89":6,
    ">=90":7
}
aRango=["<=39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    ">=90"
]

if opcion_nav=="Nacional":  
    opcion_rango = st.sidebar.selectbox('Elegir rango',aRango)

if opcion_nav=="Región":
    opcion_region = st.sidebar.selectbox('Elegir region',aRegiones)

#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

dOpciones={"Comuna":["Casos confirmados","Fallecidos"],"Región":["Casos Sintomaticos","Casos Asintomaticos","Fallecidos Diario"],"Nacional":["Fallecidos por rango Etario","Media Movil de Casos Nuevos por 100,000Hab"]}

option = st.selectbox('Elegir',dOpciones[opcion_nav])
st.write('Gráfico seleccionado:', option)
st.subheader(option)
if opcion_nav=="Región":
    st.write('Region Selecionada: %s'%(opcion_region))
if opcion_nav=="Nacional":
    st.write('Rango Seleccionado: %s'%(opcion_rango))
################  GRAFICO #################
if st.checkbox('Generar gráfico'):
    if opcion_nav == "Comuna":
        if option == "Casos confirmados":
            #from  import prediccion
            #st.line_chart(prediccion)
            st.title('Falta aun')
        elif option == "Fallecidos":
            #from  import prediccion
            #st.line_chart(prediccion)
            st.title('Falta aun')
    elif opcion_nav == "Región":
        if option == "Casos Sintomaticos":
            import Redes_neuronales_casos_region_sintomaticos as cR_S
            #from Redes_neuronales_casos_region_sintomaticos import prediccion
            st.line_chart(cR_S.busca(int(dDic_regiones[opcion_region])))
        elif option == "Casos Asintomaticos": 
            import Redes_neuronales_casos_region_asintomaticos as cR_A
            #from Redes_neuronales_casos_region_asintomaticos import prediccion
            st.line_chart(cR_A.busca(int(dDic_regiones[opcion_region])))
        elif option == "Fallecidos Diario": 
            import Redes_neuronales_Region_fallecidos as cR_F
            #from Redes_neuronales_Region_fallecidos import prediccion
            st.line_chart(cR_F.busca(int(dDic_regiones[opcion_region])))
    elif opcion_nav == "Nacional":
        if option == "Fallecidos por rango Etario":
            import Redes_neuronales_fallecidos_nacional as fN
            st.line_chart(fN.busca(int(dDic_rango[opcion_rango])))
        elif option == "Media Movil de Casos Nuevos por 100,000Hab":
            from Redes_neuronales_casos_nacional import prediccion
            st.line_chart(prediccion)

    

###########################################





st.write('Autores: Marco Pedemonte, Juan Pérez, Ignacio Ringler, Nicolás Rivas')