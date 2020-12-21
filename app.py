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

#options = st.sidebar.multiselect('Elegir regiones',['1', '2', '3', '4'],['1', '3'])
#st.sidebar.write('You selected:', options)

#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

dOpciones={"Comuna":["Casos confirmados","Fallecidos"],"Región":["Casos Sintomaticos","Casos Asintomaticos","Fallecidos Diario"],"Nacional":["Fallecidos por rango Etario","Media Movil de Casos Nuevos por 100,000Hab"]}

option = st.selectbox('Elegir',dOpciones[opcion_nav])
st.write('Gráfico seleccionado:', option)
st.subheader(option)

################  GRAFICO #################
if st.checkbox('Show dataframe'):
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
            from Redes_neuronales_casos_region_sintomaticos import prediccion
            st.line_chart(prediccion)
        elif option == "Casos Asintomaticos": 
            import Redes_neuronales_casos_region_asintomaticos as cR_A
            #from Redes_neuronales_casos_region_asintomaticos import prediccion
            opcion=1
            st.line_chart(cR_A.busca(opcion))
        elif option == "Fallecidos Diario": 
            from Redes_neuronales_Region_fallecidos import prediccion
            st.line_chart(prediccion)
    elif opcion_nav == "Nacional":
        if option == "Fallecidos por rango Etario":
            #from Redes_neuronales_casos_region_asintomaticos import prediccion
            #st.line_chart(prediccion)
            st.title('Falta aun')
        elif option == "Media Movil de Casos Nuevos por 100,000Hab":
            from Redes_neuronales_casos_nacional import prediccion
            st.line_chart(prediccion)

    

###########################################




st.write('Autores: Marco Pedemonte, Juan Pérez, Ignacio Ringler, Nicolás Rivas')