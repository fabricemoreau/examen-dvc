import os
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

print(joblib.__version__)

X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_test = np.ravel(y_test)

rf_classifier = joblib.load('models/trained_model.pkl')

# prediction
y_test_pred = rf_classifier.predict(X_test)

# save predictions
if os.path.exists('data/predictions') == False:
    os.makedirs('data/predictions')
pd.DataFrame(y_test_pred, columns=['predictions']).to_csv('data/predictions/y_test_pred.csv', index=False)

# evaluate model
metrics = {
    'mean_squared_error': mean_squared_error(y_test, y_test_pred),
    'mean_absolute_error': mean_absolute_error(y_test, y_test_pred),
    'r2_score': r2_score(y_test, y_test_pred)
}
pd.DataFrame(metrics, index=[0]).to_json('metrics/scores.json', orient='records', lines=True)
print("Model evaluation completed and metrics saved successfully.")
