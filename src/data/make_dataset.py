import os
from check_structure import check_existing_file
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw_data/raw.csv"
PROCESSED_DATA_PATH = "data/processed_data"

df = pd.read_csv(RAW_DATA_PATH, parse_dates=['date'])
# on considère que l'horaire et les saisons n'ont pas d'impact sur le processus de fabrication, on supprime la colonne date
df = df.drop(columns=['date'])
X = df.drop(columns=['silica_concentrate'])
y = df['silica_concentrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# sauvegarde les données
for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    output_filepath = os.path.join(PROCESSED_DATA_PATH, f'{filename}.csv')
    if check_existing_file(output_filepath):
        file.to_csv(output_filepath, index=False)