
# Script de normalisation des données:
# L’objectif de ce script est de normaliser les données à l'aide de StandardScaler

# Import librairies
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chargement des jeux de données
file_path = "data/processed_data/"
X_train = pd.read_csv(file_path + "X_train.csv")
X_test = pd.read_csv(file_path + "X_test.csv")

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des jeux de données normalisés
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(file_path + "X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(file_path + "X_test_scaled.csv", index=False)

