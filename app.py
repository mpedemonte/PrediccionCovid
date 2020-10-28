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


#################### TITULO
st.title('Predicción del covid-19 en Chile')

############# EJEMPLOS CABECERA ###############
#st.header("EJEMPLO Cabecero")
#st.subheader("EJEMPLO SubCabecero")
###############################################


opciones_grafico = ['1', '2', '3']

option = st.selectbox('Elegir gráfico',opciones_grafico)
st.write('Gráfico seleccionado:', option)

st.subheader(option)

################  GRAFICO #################
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)
###########################################


################  CHECKBOX GRAFICO
if st.checkbox('mostrar segundo gráfico'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
###########################################

st.write('Autores: Marco Pedemonte, Juan Pérez, Ignacio Ringler, Nicolás Rivas')









#########################################################
#########################################################
#########################################################
####################  SIDEBAR  ##########################
#########################################################
#########################################################
#########################################################

st.sidebar.header("Navegación")

opcion_nav = st.sidebar.radio("",('Casos', 'Defunciones Registro Civil', 'Datos Deis','Ocupación Hospitalaria','Positivad Diaria'))

options = st.sidebar.multiselect('Elegir regiones',['1', '2', '3', '4'],['1', '3'])
#st.sidebar.write('You selected:', options)

#########################################################
#########################################################




