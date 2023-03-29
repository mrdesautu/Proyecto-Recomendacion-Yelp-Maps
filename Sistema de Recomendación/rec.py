import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64
#librería de manipulación de dataframes
import pandas as pd
#Funciones matemáticas, necesitaremos sólo importar la función sqrt
from math import sqrt
#deteccion de Lemgiaje natural
import spacy
#Implementacion de un generador de texto predictivo añadido. 
import openai
#Importante!
#se debe ejecutar el siguiente comando para descargar la librereia(!python -m spacy download es_core_news_sm)
ratings_df = pd.read_parquet('Y_rating_CA')

image = Image.open('images (1).jpg')
st.title('Chef Kristoph')
st.image(image, caption='¡Disfruta tu experiencia culinaria!')

st.markdown('***')

st.markdown('## ¿Qué hará Chef Kristoph por mi?')
st.markdown(''' Chef Kristoph es una I.A. que con los filtros más rigurosos para solucionar dos situaciones:''')
st.markdown('1- Te recomendará, no solo el mejor lugar pensado para ti, también te dira y mostrará el por qué lo pensó!''')
st.markdown('2- Decidirá por ti entre opciones,  te mostrará y te explicará el por qué!')
st.markdown( '''¡Increible! Pensar donde comer ya no será un problema''')
st.markdown('***')
text_input = st.text_input(
        "Ingrese su nombre 👇")
#preferencia_usuario = st.text_input("Ingrese sus preferencias")
option = st.selectbox(
    'SELECCIONE SU NUMERO DE USUARIO',
    ('Ninguno','OVLf6NVTi7noMP1qCKr76w','B5s_DCLVrBLrL8U6TEVlwA', 'hntJTn1Ev6TXfusbvTdutw', 'o6UJMpHcpLJEvmKLrxLS3w'))

st.write('usuario numero:', option)

def get_user(usuario_ingresado):
    userp = ratings_df[ ratings_df['user_id'] == usuario_ingresado]
    return userp
userp= get_user(option)
#Filtrando los usuarios que han ido a restaurantes y guardándolas
inputMovies=userp
userSubset = ratings_df[ratings_df['business_id'].isin(inputMovies['business_id'].tolist())]
userSubsetGroup = userSubset.groupby(['user_id'])
#Ordenamiento de forma tal de que los usuarios con más películas en común tengan prioridad
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup = userSubsetGroup[0:1000]
pearsonCorrelationDict = {}

# Compute sum of squares for inputMovies ratings
inputMoviesSumSquares = sum([rating**2 for rating in inputMovies['stars_y']])

for name, group in userSubsetGroup:
    group = group.sort_values(by='business_id')
    inputMovies = inputMovies.sort_values(by='business_id')

    nRatings = len(group)

    # Compute sum of squares and sum of products for group ratings
    groupSumSquares = sum([rating**2 for rating in group['stars_y']])
    sumProducts = sum(a*b for a,b in zip(inputMovies['stars_y'], group['stars_y']))

    # Compute Pearson correlation coefficient
    denominator = sqrt(inputMoviesSumSquares * groupSumSquares)
    if denominator != 0:
        pearsonCorrelationDict[name] = sumProducts / denominator
    else:
        pearsonCorrelationDict[name] = 0
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['user_id'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
#Ahora obtengamos los 50 primeros usuarios más parecidos a los que se ingresaron.
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsersRating=topUsers.merge(ratings_df, left_on='user_id', right_on='user_id', how='inner')
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['stars_y']
#Se aplica una suma a los topUsers luego de agruparlos por userId
tempTopUsersRating = topUsersRating.groupby(topUsersRating['business_id']).sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns =['sum_similarityIndex','sum_weightedRating']
#Se crea un dataframe vacío
recommendation_df = pd.DataFrame()
#Ahora se toma el promedio ponderado
recommendation_df['weighted_average_recommendation_score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
#Luego, ordenémoslo y veamos las primeras películas que el algoritmo recomendó!
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df['business_id'] = recommendation_df.index
# Ordenar los valores por la columna "weighted_average_recommendation_score" de manera descendente y seleccionar los 5 primeros registros
df_top5 = recommendation_df.nlargest(5, 'weighted_average_recommendation_score')
df_top5 = df_top5.reset_index(drop=True)
df_r =pd.read_parquet("Y_business_CA")
df_review =pd.read_parquet("Y_review_CA")
df_user =pd.read_parquet("Y_user_CA")
df_merged = pd.merge(df_top5, df_r, on='business_id' )
df_merged = pd.merge(df_merged, df_review, on='business_id')
df_merged = df_merged.iloc[:, [1, 2, 4, 9, 13, ] + list(range(16, len(df_merged.columns)))]
df_merged = pd.merge(df_merged, df_user, on='user_id')
# Obtener los usuarios con más seguidores para cada valor en la columna "id"
df_top_users = df_merged.groupby('business_id').apply(lambda x: x.nlargest(1, 'fans'))
df = df_top_users[['name_x', 'city', 'stars', 'categories', 'text']]
valor1 = str(df.iloc[0, 0])
valor2 = str(df.iloc[0, 1])
valor3 = str(df.iloc[0, 3])
valor4 = str(df.iloc[0, 4])
valor5 = str(df.iloc[1, 0])
valor6 = str(df.iloc[1, 1])
valor7 = str(df.iloc[1, 3])
valor8 = str(df.iloc[1, 4])
valor9 = str(df.iloc[2, 0])
valor10 = str(df.iloc[2, 1])
valor11 = str(df.iloc[2, 3])
valor12 = str(df.iloc[2, 4])
nombre= str(text_input)
preferencias= "ninguna"
#prompt_sugerencia= f"Actúa como critico culinario, genera un mensaje para {nombre}, con estas preferencias: {preferencias} 1-nombre:{valor1}, ciudad:{valor2}, categorias:{valor3}, otro usuario lo ha recomendado diciendo:{valor4}, responde en español, debes mencionar el nombre, la ciudad y una oracion explicando porqué visitar el lugar "
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
prompt_ingles= f"Act as a food critic and generate a message for {nombre} with the following preferences: {preferencias}. The restaurant's name is {valor1}, located in {valor2}, and it falls under the categories of {valor3}. Another user has recommended it, saying: {valor4}. Respond in Spanish and mention the restaurant's name, location, and why someone should visit."
respuesta1= generar_respuesta(prompt_ingles)
#print(f"Chef Kristoph: {respuesta1}")
#respuesta3= generar_respuesta(prompt_ingles)
#print(f"Chef Kristoph: {respuesta3}")
prompt_eleccion= f"Choose between 2 restaurants: 1-{valor9}, located in {valor10}, categorized as {valor11} and it has been said {valor12} or 2-{valor5}, located in {valor6}, categorized as {valor7} and it has been said {valor8}. Which one do you think is better for {nombre} and why? Respond in Spanish."
respuesta2= generar_respuesta(prompt_eleccion)

# Cargar el modelo en español
nlp = spacy.load("es_core_news_sm")
#se debe modificar por respuesta 2 para graficar la elección. 
texto = respuesta1
texto1 = respuesta2
#texto2 = respuesta3
# Definir una lista de términos relacionados con la comida
terminos_comida = ["comida"," pretzel","comidas latinoamericanas","platillo de pierna de rana","margarita","familia", "amigos" ,"papa" ,"fries","salsa","fritas", "plato", "cena", "desayuno", "almuerzo", "restaurante", "pizza", "pasta", "italiana", "mariscos", "carne", "pollo", "vegetariana","abadejo", "aceituna", "achicoria", "acompañamiento", "acónito", "adobo", "agua", "aguacate", "ahumado", "ajo", "albaricoque", "albóndiga", "alcachofa", "alcaparra", "alcohol", "ale", "alfajor", "alga", "algarrobo", "almeja", "almendra", "alubia", "alubias blancas", "alubias rojas", "alubias verdes", "amapola", "amaranto", "anacardo", "anchoa", "anís", "apio", "arándano", "aroma", "arracacha", "arroz", "arándano", "asafoetida", "asado", "atún", "avena", "avellana", "azúcar", "bacalao", "bacón", "bagel", "baileys", "banana", "barbacoa", "barquillo", "basmati", "batata", "bayas", "bechamel", "bebida", "bebida energética", "bebida isotónica", "bebida vegetal", "berenjena", "berro", "besugo", "betabel", "biberón", "bicarbonato", "bistec", "bizcocho", "boletus", "boniato", "borscht", "brioche", "brocheta", "brocoli", "brócoli", "brownie", "buey", "búfalo", "bulgur", "buttercream", "cacahuete", "cacao", "café", "caipirinha", "cajeta", "calabacín", "calamares", "caldo", "callos", "calzone", "camarones", "canela", "cangrejo", "canguil", "cantaloupe", "caperuza", "carambola", "caramelo", "carbonada", "cardamomo", "carnaval", "carnaza", "carne", "carne de cerdo", "carne de res", "carne de vaca", "carne molida", "carne picada", "carne seca", "carnitas", "carnitas", "carpaccio", "carpa", "carpaccio", "casa", "casabe", "caserola", "castañas", "causa", "cebada", "cebiche", "cebolla", "cebolla morada", "cecina", "cecina de res", "cecina de yecapixtla", "cecina enchilada", "cecina enchilada", "ceja", "celebración", "cena", "cerdo", "cereales", "cereza", "cerveza", "cerveza oscura", "chabacano", "chalupa", "champán", "champiñón", "champiñones", "chancho", "chayote", "cheesecake", "chía", "chicha", "chicle", "chicharrón", "chicharrón de cerdo", "chicharrón de queso", "chicken nuggets", "ensalada", "sopa", "gazpacho", "cebiche", "guacamole", "ceviche", "empanada", "taco", "burrito", "quesadilla", "nachos", "chimichanga", "enchilada", "tamal", "arepa", "patacones", "yuca", "arroz", "frijoles", "tostones", "plátano", "plátano maduro", "maíz", "carnitas", "chicharrón", "chorizo", "tortilla", "gordita", "huevo", "queso", "chile", "ajiaco", "aji", "mole", "cuitlacoche", "tlacoyo", "huitlacoche", "pipian", "pollo a la brasa", "aji de gallina", "seco de res", "lomo saltado", "ajiaco", "aji de polleria", "anticuchos", "caldo de gallina", "carapulcra", "causa rellena", "ceviche de pescado", "chicharron de pescado", "chupe de camarones", "jalea mixta", "ocopa", "papa a la huancaína", "papa rellena", "rocoto relleno", "sopa criolla", "tacu tacu", "tallarines verdes", "tiradito", "trucha frita", "trucha a la parrilla", "rocoto", "rocoto relleno", "sillao", "seco de cabrito", "seco de chabelo", "pato en maní", "chilcano", "lomo fino", "lagarto de res", "churrasco", "picante de cuy", "chicha morada", "pisco sour", "causa a la limeña", "chilcano de pisco", "ajiaco de olluco", "ají de papas", "carne a la piedra", "cau cau", "cuy chactado", "chuleta de cerdo", "chicharrón de cerdo", "ajiaco de papas", "locro de papas", "ensalada rusa", "empanadas de verde", "fanesca", "hornado", "hornado de chancho", "lechón", "lojano", "mote pata", "mote pillo", "pepian de choclo", "porotos", "seco de chivo", "tigrillo", "trigo mote", "zambo", "caldo de pescado", "causa de atún", "jalea de mariscos", "arroz chaufa", "chancho al palo", "chifa", "churrasco a lo pobre", "pan con chicharrón", "papa rellena", "papa a la huancaína", "rocoto relleno", "sancochado", "sudado de pescado", "sudado de machas", "sudado de mariscos", "tamales verdes", "tamales de chanchamito", "tamales de camarones", "tallarines con bistec", "tallarines con pollo","arroz", "fideos", "sopas", "caldos", "guisos", "estofados", "carnes", "pescados", "mariscos", "pollo", "cerdo", "vacuno", "ternera", "cordero", "chuletas", "filetes", "asados", "ensaladas", "verduras", "legumbres", "frutas", "postres", "tartas", "pasteles", "helados", "bebidas", "refrescos", "cerveza", "vino", "licores", "cafés", "tés", "chocolates", "desayunos", "meriendas", "aperitivos", "tapas", "embutidos", "quesos", "hamburguesas", "hot dogs", "sushi", "rolls", "kebab", "tacos", "burritos", "pizzas", "hamburguesas veganas", "batidos", "smoothies", "zumos", "milkshakes", "patatas fritas", "nachos", "alitas de pollo", "guacamole", "ceviche", "tortillas", "enchiladas", "quesadillas", "tamales", "chorizo", "morcilla", "fajitas", "chimichangas", "chiles rellenos", "gazpacho", "salmorejo", "paella", "tortilla española", "calamares a la romana", "croquetas", "bacalao", "sardinas", "churros", "flan", "leche frita", "torrijas", "pulpo a la gallega", "cocido madrileño", "fabada asturiana", "gazpachos manchegos", "rabas", "piparras", "gulas", "ensaimadas", "sobao pasiego", "quesada pasiega", "tarta de queso", "rosquillas", "mantecados", "polvorones", "panettone", "sobaos", "cheesecake", "brownies", "cupcakes", "donuts", "waffles", "crepes", "tiramisu", "macarons", "cookies", "chips ahoy", "oreos", "toblerone", "kitkat", "mars", "snickers", "hershey's", "milky way", "twix", "fanta", "sprite", "coca-cola", "pepsi", "seven up", "mountain dew", "red bull", "monster energy", "gatorade", "agua", "zumo de naranja", "zumo de manzana", "zumo de piña", "zumo de tomate", "smoothie de frutas", "batido de chocolate", "capuccino", "café con leche", "café americano", "café expreso", "té verde", "té negro", "té chai", "té de jazmín", "chocolate caliente", "salsa barbacoa", "salsa de queso", "salsa de tomate", "salsa rosa", "alioli", "mayonesa", "mostaza", "ketchup", "salsa tártara"]

# Procesar el texto con spaCy
doc = nlp(texto)

# Recorrer todas las entidades del documento y obtener las que contengan términos de comida
comida = []
for token in doc:
    if token.lower_ in terminos_comida:
        comida.append(token.text)

# Procesar el texto con spaCy
doc1 = nlp(texto1)

# Recorrer todas las entidades del documento y obtener las que contengan términos de comida
comida1 = []
for token in doc1:
    if token.lower_ in terminos_comida:
        comida1.append(token.text)
def generar_imagen(descripcion):
    response = openai.Image.create(
                                    prompt=descripcion,
                                    n=1,
                                    size="256x256",
                                    response_format="b64_json"
                                    )
    for i in range(0,len(response['data'])):
        b64=response['data'][i]['b64_json']
        with open(f'image_{i}.pgn', 'wb') as f:
            f.write(base64.urlsafe_b64decode(b64))
        return response
    #return response
img= (generar_imagen(str(comida)))
valor = img['data'][0]['b64_json']
# Decodificar la imagen desde la cadena base64
b64_string = valor
image_data = base64.b64decode(b64_string)
# Convertir los datos de la imagen en un objeto de imagen
img = Image.open(BytesIO(image_data))
st.markdown('***')

st.markdown('# ¿ Qué me recomienda Chef Kristoph?')
st.markdown('## Opción predilecta')
image = Image.open('images (2).jpg')
st.image(image)
st.markdown(f'''Chef Kristoph: {respuesta1}''')
st.image(img)

st.markdown('***')

st.markdown('# ¡Ayudame a elegir!')
st.markdown('## Elegirá entre la 2° y 3° opción!')
image = Image.open('images (2).jpg')
st.image(image)
st.markdown(f'''Chef Kristoph: {respuesta2}''')
img1= (generar_imagen(str(comida1)))
valor1 = img1['data'][0]['b64_json']
# Decodificar la imagen desde la cadena base64
b64_string = valor1
image_data1 = base64.b64decode(b64_string)
# Convertir los datos de la imagen en un objeto de imagen
img1 = Image.open(BytesIO(image_data1))
st.image(img1)
st.markdown('***')
st.title(' _Bon Appétit_ ')
image = Image.open('white-friendly-chef-robot-with-pan-and-turner-cooking-robot-artificial-intelligence-concept-vector.jpg')
st.image(image)
st.markdown('***')