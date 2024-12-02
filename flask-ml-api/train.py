from trainingmodel.preprocess import dataframe_process
from trainingmodel.rawData import diccionario
from trainingmodel.models import split_data, train_model

df = dataframe_process(diccionario)


x_train, x_test, y_train, y_test = split_data(df)

train_model(x_train, y_train, x_test, y_test)