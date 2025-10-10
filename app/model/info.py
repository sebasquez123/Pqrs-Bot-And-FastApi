best_params = {'C': 1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}
test_accuracy = 0.68
train_accuracy = 1.0
classification_report = '''                     precision    recall  f1-score   support

     Business hours       1.00      1.00      1.00         2
              Claim       1.00      1.00      1.00         1
          Complaint       0.33      1.00      0.50         1
Contact information       1.00      0.67      0.80         3
           Farewell       0.33      0.67      0.44         3
           Greeting       0.50      1.00      0.67         1
             Prices       1.00      1.00      1.00         2
            Request       0.50      0.50      0.50         2
      Return policy       1.00      0.80      0.89         5
  Technical support       1.00      0.20      0.33         5

           accuracy                           0.68        25
          macro avg       0.77      0.78      0.71        25
       weighted avg       0.83      0.68      0.68        25
''' 
confusion_matrix = '''[[2 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0]
 [0 0 0 2 1 0 0 0 0 0]
 [0 0 0 0 2 1 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 2 0 0 0]
 [0 0 1 0 0 0 0 1 0 0]
 [0 0 1 0 0 0 0 0 4 0]
 [0 0 0 0 3 0 0 1 0 1]]''' 
