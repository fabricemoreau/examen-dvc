import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

print(joblib.__version__)

PROCESSED_DATA_PATH = "data/processed"

X_train = pd.read_csv(PROCESSED_DATA_PATH + '/X_train_scaled.csv')
y_train = pd.read_csv(PROCESSED_DATA_PATH + '/y_train.csv')
y_train = np.ravel(y_train)

param_rf = {
    'n_estimators': [50, 100, 200, 400], 
    'max_depth': [None, 5, 10, 15],
    'max_features': range(1, X_train.shape[1])
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(n_jobs = -1, random_state = 42), 
                           param_grid=param_rf, 
                           cv=3, 
                           n_jobs = -1, 
                           verbose = 1)
grid_search.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/best_model.pkl'
joblib.dump(grid_search.best_estimator_, model_filename)
print("Best model found and saved successfully.")
