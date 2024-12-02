import joblib
import spacy
from models.info import best_params, test_accuracy, train_accuracy

nlp = spacy.load("es_core_news_md")


def data_treatment(data):
    vectorizer = joblib.load("models/vectorizer.pkl")
    doc = nlp(data)  
    pregunta_procesada=" ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    pregunta_bagword = vectorizer.transform([pregunta_procesada]).toarray()
    return pregunta_bagword
    

def predict_model(input_data):
    
    input= data_treatment(input_data)
    model = joblib.load("models/modelv1.pkl")
    output = model.predict(input)
    print(f"La clase predicha para la pregunta es: {output[0]}")
    return output[0]
    # return f"La pregunta ingresada es: {input_data}"
def model_info():
    return  best_params, test_accuracy, train_accuracy