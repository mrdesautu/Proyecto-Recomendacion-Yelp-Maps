import math
import pandas as pd
import streamlit as st
#Implementacion de un generador de texto predictivo añadido. 
import openai
from PIL import Image
import requests
from io import BytesIO
import base64
import pyttsx3
import tempfile



image = Image.open('robot inversor.jpg')
st.title('RoboAdvisor')
st.image(image, caption='¡Ideas Inteligentes!')

st.markdown('***')

df = pd.read_parquet("Y_business_CA")
st.markdown('## ¿Cómo funciona RoboAdvisor?')
st.markdown(''' RoboAdvisor es una I.A. que con los sistemas mas invovadores, ayuda a los inversionistas a crear ideas basadas en datos.''')
st.markdown(''' El sistema aprendera de las fortalezas y debilidades de otros retaurantes de categorias identicas a las que el cliente decida.''')
st.markdown(''' Contemplará cercanía y aspectos geograficos.''')
st.markdown( '''¡Encuentra tu oportunidad de inversión!''')

option_id = st.selectbox("Seleccione el Id del Restaurante a comparar, lo central aquí es la ubicación",("IDtLPgUrqorrpqSLdfMhZQ","VeFfrEZ4iWaecrQg6Eq4cg","4xhGQGdGqU60BIznBjqnuA","uFF40n9pOqHK1ciajdoSEw", "VeFfrEZ4iWaecrQg6Eq4cg"))

id_seleccionado = option_id
latitudes_seleccionadas = df.loc[df['business_id'] == id_seleccionado , 'latitude']
longitudes_seleccionadas = df.loc[df['business_id'] == id_seleccionado , 'longitude']
lat_ref = latitudes_seleccionadas
lon_ref = longitudes_seleccionadas
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radio de la Tierra en kilómetros
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d
df['distancia'] = df.apply(lambda row: haversine(lat_ref, lon_ref, row['latitude'], row['longitude']), axis=1)
df = df.sort_values(by=['distancia'])
df_cercanos = df.head(100)
top_cercanos=df_cercanos.nlargest(10, 'stars')
top_opciones=top_cercanos['categories'].unique().astype(str)
opciones1=top_opciones
low_cercanos = df_cercanos.nsmallest(10, 'stars')
low_opciones=low_cercanos['categories'].unique().astype(str)
opciones2= low_opciones
optiontop = st.selectbox(
    'Seleccione rubros',
    (opciones1))
# Definir variable filtro
filtro_top = optiontop

# Aplicar filtro solo si se proporciona una palabra
if filtro_top:
    data_top = df_cercanos[df_cercanos['categories'].str.contains(filtro_top, case=False)]
    

optionlow = st.selectbox(
    'Seleccione rubros',
    (opciones2))
# Definir variable filtro
filtro_low = optionlow

# Aplicar filtro solo si se proporciona una palabra
if filtro_low:
    data_low = df_cercanos[df_cercanos['categories'].str.contains(filtro_low, case=False)]
    
df_tips = pd.read_parquet("Y_tips_CA")
merged_df_top = data_top.merge(df_tips, on='business_id')
merged_df_low = data_low.merge(df_tips, on='business_id')

openai.api_key = "" # nunca dejen sus credenciales dentro del código! pueden hacerlo una variable de ambiente

def generar_respuesta(prompt, modelo="text-davinci-003", temperatura=1, max_tokens=300, top_p=1, frecuencia_penalizacion=0, presencia_penalizacion=0):
    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        temperature=temperatura,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frecuencia_penalizacion,
        presence_penalty=presencia_penalizacion
    )

    return respuesta.choices[0].text

comentarios_top= merged_df_top['text'].astype(str)
comentarios_low= merged_df_low['text'].astype(str)
ciudad_top=merged_df_top['city'].iloc[0]
ciudad_low=merged_df_low['city'].iloc[0]
prompt_top = f"Act as an investment advisor and provide 3 ideas to improve a restaurant in the {optiontop} industry located in the city of {ciudad_top}. Some of the best places in the city have received positive comments about {comentarios_top}. Based on this information, what recommendations would you give to someone who wants to open a restaurant? Please respond in Spanish."
prompt_low = f"Act as an investment advisor and generate 3 ideas to improve a restaurant in the {optionlow} industry located in the city of {ciudad_low}. Some of the worst places in the city have received comments to learn from: {comentarios_low}. Learn from the mistakes mentioned. What recommendations would you give to someone who wants to open a restaurant? Please respond in Spanish."
respuesta_top= generar_respuesta(prompt_top)
respuesta_low= generar_respuesta(prompt_low)


st.markdown('***')
st.markdown( f'''# Aprendizaje sobre las fortalezas de otros Restaurantes de {ciudad_top}''')
#imagen
image = Image.open('images (4).jpg')
st.image(image)
#respuesta
st.markdown(f'''RoboAdvisor: {respuesta_top}''')


st.markdown('***')
st.markdown( f'''# Aprendizaje sobre las debilidades de otros Restaurantes de {ciudad_low} ''')
#imagen
image = Image.open('images (5).jpg')
st.image(image)
#respuesta
st.markdown(f'''RoboAdvisor: {respuesta_low}''')

#text_input = st.text_input(" preferencias para un logo exitoso")
st.markdown('***')
st.markdown( f'''# ¡Haga una diferencia en {ciudad_low}! ''')

image = Image.open('images (6).jpg')
st.image(image)


