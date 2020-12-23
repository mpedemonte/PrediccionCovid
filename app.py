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
from app_dic_array import *

st.title('Predicción del covid-19 en Chile')

st.sidebar.header("Navegación")

opcion_nav = st.sidebar.radio("",('Comuna', 'Región','Nacional'))


if opcion_nav=="Nacional":  
    opcion_rango = st.sidebar.selectbox('Elegir rango',aRango)

if opcion_nav=="Región":
    opcion_region = st.sidebar.selectbox('Elegir region',aRegiones)

if opcion_nav == "Comuna":
    opcion_region = st.sidebar.selectbox('Elegir region',aRegiones)
    if opcion_region == "Arica y Parinacota":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComArica)
    if opcion_region == "Tarapacá":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComTarapaca)
    if opcion_region == "Antofagasta":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComAntofagasta)
    if opcion_region == "Atacama":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComAtacama)
    if opcion_region == "Coquimbo":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComCoquimbo)
    if opcion_region == "Valparaíso":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComValparaiso)
    if opcion_region == "Metropolitana":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComMetropolitana)
    if opcion_region == "O’Higgins":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComOhiggins)
    if opcion_region == "Maule":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComMaule)
    if opcion_region == "Ñuble":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComNuble)
    if opcion_region == "Biobío":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComBiobio)
    if opcion_region == "Araucanía":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComAraucania)
    if opcion_region == "Los Ríos":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComRios)
    if opcion_region == "Los Lagos":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComLagos)
    if opcion_region == "Aysén":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComAysen)
    if opcion_region == "Magallanes":
        opcion_comuna = st.sidebar.selectbox('Elegir Comuna',aComMagallanes)

dOpciones={"Comuna":["Casos confirmados","Fallecidos"],"Región":["Casos Sintomaticos","Casos Asintomaticos","Fallecidos Diario"],"Nacional":["Fallecidos por rango Etario","Media Movil de Casos Nuevos por 100,000Hab"]}

option = st.selectbox('Elegir',dOpciones[opcion_nav])
st.write('Gráfico seleccionado:', option)
st.subheader(option)
if opcion_nav=="Región":
    st.write('Region Selecionada: %s'%(opcion_region))
if opcion_nav=="Nacional":
    st.write('Rango Seleccionado: %s'%(opcion_rango))
if opcion_nav == "Comuna":
    st.write('Rango Seleccionado: %s'%(opcion_comuna))
################  GRAFICO #################
if st.checkbox('Generar gráfico'):
    if opcion_nav == "Comuna":
        if option == "Casos confirmados":
            import Redes_neuronales_casos_comuna as cC
            st.line_chart(cC.busca(int(dDic_comunas[opcion_comuna])))
        elif option == "Fallecidos":
            import Redes_neuronales_fallecidos_comuna as fC
            st.line_chart(fC.busca(int(dDic_comunas[opcion_comuna])))
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

st.write('Autores: Marco Pedemonte, Juan Pérez, Ignacio Ringler, Nicolás Rivas')