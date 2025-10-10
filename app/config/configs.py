
import os

# Configuration for model and vectorizer paths and names
model_name = "model_v1.pkl"

vectorized_name = "vectorizer.pkl"

app_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(app_dir, "model")

## hyperparameter search space for GridSearchCV
param_grid = {  
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
            'solver': ['liblinear', 'lbfgs'],  # Optimization algorithms
            'max_iter': [100, 500, 1000],  # Maximum number of iterations
            'penalty': ['l2', 'l1'],  # Type of regularization
        }

