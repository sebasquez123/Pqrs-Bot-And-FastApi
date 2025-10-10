import logging
from app.model.raw import dictionary
from app.training.vectorizer import make_dataframe
from app.training.train import split_data, train_model


def workshop():
    print("""
=================================================================
Attention! 

You are about to start the Pqrs-Bot training process.
This script prepares and transforms the data, splits it into training
and test sets, and trains a lightweight model designed for production
with basic functionality that can scale to more complex landscapes.

Make sure you have the dependencies installed in a virtualenv (see `requirements.txt`),
and that your data is correctly formatted in the `raw.py` file before continuing.

If you want to cancel, press
Ctrl+C 
=================================================================
""")
    try:
        input("Would you like to continue? Press Enter to proceed or Ctrl+C to cancel...")
        # Prodeceding with data processing and dataframe creation.
        df = make_dataframe(dictionary)
        # splitting the data into training and test sets.
        div_data = split_data(df)
        # training the model with the split data.
        train_model(*div_data)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

