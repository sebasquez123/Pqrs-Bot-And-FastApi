import pandas as pd
from app.model.info import best_params, test_accuracy, train_accuracy,classification_report,confusion_matrix
from app.config.configs import  model_path, vectorizer_path
import joblib
import spacy



class BotService:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def predict_model(self, input_data):
        '''Predict the class for the given input data.'''
        
        try:
            bagwords_input = self._data_tokenizer(input_data)
            model = joblib.load(model_path)
            if not model:
                raise RuntimeError("Model not found. Please ensure the model is trained and available.")
            output = model.predict(bagwords_input)
            
            return output[0]
        
        except Exception as e:
            raise RuntimeError(str(e))

    def model_info(self):
        '''Fetch and return model information.'''
        
        return best_params, test_accuracy, train_accuracy,classification_report,confusion_matrix

    def _data_tokenizer(self, data):
        '''Preprocess the input data for prediction, including tokenization, lemmatization, and vectorization.'''
        
        try:
            
            vectorizer = joblib.load(vectorizer_path)
            if not vectorizer:
                raise RuntimeError("Vectorizer not found. Please ensure the vectorizer is trained and available.")
            doc = self.nlp(data)  
            processed_data = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
            bagword = vectorizer.transform([processed_data]).toarray()
            df_bagword = pd.DataFrame(bagword, columns=vectorizer.get_feature_names_out())

            return df_bagword
        
        except Exception as e:
            raise e
        
    