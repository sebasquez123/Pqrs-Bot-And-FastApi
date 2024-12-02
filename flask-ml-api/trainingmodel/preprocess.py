import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Cargar el modelo de idioma español de spaCy
nlp = spacy.load("es_core_news_md")

#SPACY PARA TOKENIZAR, LIMPIAR, LEMATIZAR Y ELIMINAR STOPWORDS
def transformar_frase(frase):
    doc = nlp(frase)  
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


def dataframe_process(diccionario):

    data = []

    #DEVUELVE UN ARREGLO PLANO CON CADA UNA DE LAS FRASES DENTRO DE LOS VALUES.    
    for categoria, frases in diccionario.items():
        for frase in frases:
            texto_procesado = transformar_frase(frase)
            data.append([texto_procesado, categoria])

    data = np.array(data)
    # print(data)
    df = pd.DataFrame(data, columns=["Texto transformado", "Clase"])
    vacios= df[df['Texto transformado']=='']['Texto transformado'].index.tolist()
    df.drop(vacios,axis=0,inplace=True)
    df.reset_index(drop=True,inplace=True)
    # CREAMOS EL VECTORIZADOR PARA LA BOLSA DE PALABRAS
    vectorizer = CountVectorizer() 

    # ALIMENTAR EL VECTORIZADOR Y CONVERTIR LA LISTA DE PALABRAS EN UNA MATRIZ DE FRECUENCIAS (BOLSA DE PALABRAS)
    bagwords=vectorizer.fit_transform(df['Texto transformado']).toarray()
    # SE DEBE GUARDAR EL VECTORIZADOR PARA PODER USARLO EN EL SERVICIO
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    
    # CREAR EL DATAFRAME
    df_bagword = pd.DataFrame(bagwords, columns=vectorizer.get_feature_names_out())

    # SE AGREGA LA COLUMNA DE ETIQUETAS
    df_bagword['Clase'] = df['Clase']


    print("Vectorizador guardado como vectorizer.pkl")
  # df_bagword.head()
    # print(df_bagword['Clase'].value_counts())
    return df_bagword

  