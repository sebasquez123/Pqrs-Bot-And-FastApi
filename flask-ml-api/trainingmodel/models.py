from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def split_data(df_bagword):
    x= df_bagword.drop(['Clase'],axis=1)
    y= df_bagword['Clase']  
    # print('test: ',y_test.unique(),'\n')
    # print('train: ',y_train.unique())
    return train_test_split(x, y, test_size=0.2, random_state=1)


def train_model(x_train, y_train, x_test, y_test):
# Definir los parámetros a ajustar
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Parámetro de regularización
        'solver': ['liblinear', 'lbfgs'],  # Algoritmos de optimización
        'max_iter': [100, 500, 1000],  # Número máximo de iteraciones
        'penalty': ['l2', 'l1'],  # Tipo de regularización
    }

    # Crear el modelo base
    model = LogisticRegression(random_state=1)

    # Definir GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Ajustar el modelo con los mejores parámetros encontrados
    grid_search.fit(x_train, y_train)

    # Mejor combinación de parámetros
    print("Mejores parámetros encontrados:", grid_search.best_params_)

    # Predecir con el modelo ajustado
    best_model = grid_search.best_estimator_
    prediccion = best_model.predict(x_test)

    # Métricas de evaluación

    ### VALIDACIÓN DE SUB Y SOBRE AJUSTE ###
    y_train_pred = best_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    ### EVALUACIÓN ###
    print("Classification Report:\n", classification_report(y_test, prediccion))
    print("Confusion Matrix:\n", confusion_matrix(y_test, prediccion))
    
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    with open("models/info.py", "w") as file:
        file.write(f"best_params = {grid_search.best_params_}\n")  
        file.write(f"test_accuracy = {test_accuracy}\n")  # Escribe la precisión del test
        file.write(f"train_accuracy = {train_accuracy}\n")  # Escribe la precisión del entrenamiento
    print("Información guardada en info.py")


    # Guardar el modelo ajustado
    joblib.dump(best_model, "models/modelv1.pkl")
    print("Modelo guardado como modelv1.pkl")