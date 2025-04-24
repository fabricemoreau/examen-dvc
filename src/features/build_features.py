import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_PATH = "data/processed"


X_train_path = os.path.join(PROCESSED_DATA_PATH, 'X_train.csv')
X_train = pd.read_csv(X_train_path)

scale = StandardScaler()
X_train_scaled = pd.DataFrame(scale.fit_transform(X_train), columns=X_train.columns)
X_train_scaled_path = os.path.join(PROCESSED_DATA_PATH, 'X_train_scaled.csv')
X_train_scaled.to_csv(X_train_scaled_path, index=False)

X_test_path = os.path.join(PROCESSED_DATA_PATH, 'X_test.csv')
X_test = pd.read_csv(X_test_path)
X_test_scaled = pd.DataFrame(scale.transform(X_test), columns=X_test.columns)
X_test_scaled_path = os.path.join(PROCESSED_DATA_PATH, 'X_test_scaled.csv')
X_test_scaled.to_csv(X_test_scaled_path, index=False)
