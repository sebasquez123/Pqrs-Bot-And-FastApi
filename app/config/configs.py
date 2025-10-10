
import os

# Configuration for model and vectorizer paths and names
model_name = "model_v1.pkl"

vectorized_name = "vectorizer.pkl"

info_name = "info.py"

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_base_path = os.path.join(app_dir, "model")


info_path = os.path.join(model_base_path, info_name)

vectorizer_path = os.path.join(model_base_path, vectorized_name)

model_path = os.path.join(model_base_path, model_name)

## hyperparameter search space for GridSearchCV
param_grid = {  
            'C': [0.01, 0.1, 1, 10],  # Regularization parameter
            'solver': ['liblinear'],  # Optimization algorithms
            'max_iter': [100, 500],  # Maximum number of iterations
            'penalty': ['l1', 'l2'],  # Type of regularization
        }

