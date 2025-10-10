from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from app.config.configs import  info_path, param_grid, model_path
import logging
import joblib
import os



def split_data(df_bagword):
    '''This function splits the input DataFrame into training and testing sets.'''
    
    x= df_bagword.drop(['Class'],axis=1)
    y= df_bagword['Class'] 
 
    return train_test_split(x, y, test_size=0.2, random_state=1)


def train_model(x_train, x_test, y_train , y_test):
    '''This function trains a Logistic Regression model using GridSearchCV to find the best hyperparameters.
    It evaluates the model on both training and testing data, prints relevant metrics, and saves the trained model.'''

    try:
        logging.info("Starting model training... \n")
    
        ## Create the Logistic Regression model
        model = LogisticRegression(random_state=1)

        ## Define GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        ## Best parameter combination
        best_params = grid_search.best_params_
        logging.info(f"Best parameters found: {best_params} \n")

        ## Let's predict with the best model found.
        best_model = grid_search.best_estimator_

        ## Basic evaluation metrics
        logging.info("Compare Training and Testing Accuracy to see if the model is overfitting or underfitting. \n")
        ### What with the Over and Under fitting? ###
        y_train_predict = best_model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        logging.info(f"Train Accuracy: {train_accuracy} \n")

        y_test_predict = best_model.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        logging.info(f"Test Accuracy: {test_accuracy} \n")

        ### Classification and Confusion Matrix Reports ###
        classification_report_rpt = classification_report(y_test, y_test_predict)
        confusion_matrix_rpt = confusion_matrix(y_test, y_test_predict)
        logging.info(f"Classification Report:\n {classification_report_rpt}")
        logging.info(f"Confusion Matrix:\n {confusion_matrix_rpt}")

        if not os.path.exists(info_path):
            os.makedirs(info_path)

        with open(info_path, "w") as file:
            file.write(f"best_params = {grid_search.best_params_}\n")
            file.write(f"test_accuracy = {test_accuracy}\n")  
            file.write(f"train_accuracy = {train_accuracy}\n")
            file.write(f"classification_report = '''{classification_report_rpt}''' \n")
            file.write(f"confusion_matrix = '''{confusion_matrix_rpt}''' \n")

        logging.info(f"Training information saved to {info_path} \n")

        # Save the final best trained model and overwrite any previous version.
        joblib.dump(best_model, model_path)
        logging.info(f"Trained model saved to {model_path} \n")

    except Exception as e:
        raise type(e)(f"Error during the training process: {e}") from e
    