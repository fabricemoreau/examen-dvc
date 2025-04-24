
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_train = np.ravel(y_train)

rf_classifier = joblib.load('models/best_model.pkl')

#--Train the model
rf_classifier.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.pkl'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")
