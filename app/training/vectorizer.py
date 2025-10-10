
from sklearn.feature_extraction.text import CountVectorizer
from app.helpers.loading_animation import draw_progress
from app.config.configs import model_path, vectorized_name
import logging
import pandas as pd
import spacy
import numpy as np
import joblib
import time


# We use spacy to tokenize, clean, lemmatize and remove stopwords. 
# Select the size of the model and the language.
nlp = spacy.load("en_core_web_sm")


def transform_phrase(phrase):
    '''This function takes a phrase as input, processes it using spaCy
    to remove stopwords and punctuation, and returns the lemmatized and normalized form.'''
    
    doc = nlp(phrase)  
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


def make_dataframe(dictionary):
    '''This function takes a dictionary with categories as keys and lists of phrases as values,
    processes the phrases, creates a bag-of-words representation, and returns a DataFrame with the
    bag-of-words broken down into interpretable pieces and its categories.'''
    try:
        
        logging.info("Starting dataframe creation and vectorization process... \n")
        data = []
        
        # Extract and refine each phrase
        for (index, (category, phrases)) in enumerate(dictionary.items()):
            logging.info(f"\nProcessing category: {category} ({index + 1} of {len(dictionary)})")
            time_start = time.time()
            for phrase in phrases:
                processed_text = transform_phrase(phrase)
                data.append([processed_text, category])
            time_end = time.time()
            percentage = (index * 100) // (len(dictionary)-1)
            elapsed_time = time_end - time_start
            draw_progress(percentage, elapsed_time)
            
            
        # Convert to readable dataframe
        data = np.array(data)
        df = pd.DataFrame(data, columns=["Transformed text", "Class"])

        # Identify and remove empty rows
        empty_indices = df[df['Transformed text'] == '']['Transformed text'].index.tolist()
        
        # Clean up the dataframe with new indexes after dropping empty rows
        df.drop(empty_indices, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Let's create the vectorizer 
        vectorizer = CountVectorizer() 
        
        # Let's to feed the vectorizer with the transformed dataframe. this will turn the text into a bag of possible words.
        # We use fit_transform to learn the vocabulary and transform the text into a bag-of-words representation.
        # The capacity and accuracy of the vectorizer to cover all the possible cases, is determined by the number, and variety of words in the text data.
        bagwords = vectorizer.fit_transform(df['Transformed text']).toarray()
        
        # Save the vectorizer for future use in the prediction phase.
        
        joblib.dump(vectorizer, f"{model_path}/{vectorized_name}")
        
        # Recreate the dataframe.
        df_bagword = pd.DataFrame(bagwords, columns=vectorizer.get_feature_names_out())
        df_bagword['Class'] = df['Class']

        logging.info("Dataframe and vectorization process completed successfully.")
        return df_bagword
    
    except Exception as e:
        raise type(e)(f"Error during dataframe creation and vectorization process: {e}") from e

  