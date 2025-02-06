
# Script de split des données: 
# L’objectif de ce script est de diviser le dataset en X_train, X_test, y_train, y_test (features et target) 
# et d’enregistrer les fichiers dans le répertoire data/processed

# Import librairies
import pandas as pd
from sklearn.model_selection import train_test_split

# Import database
data = pd.read_csv("../../data/raw_data/raw.csv")
print(data.head())
X = data.drop(columns=["date", "silica_concentrate"])
y = data["silica_concentrate"]

# Split des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enregistrement/sauvegarde des données prétraitées dans des fichiers CSV pour les utiliser plus tard
file_path = "../../data/processed_data/"
X_train.to_csv(file_path + "X_train.csv", index=False)
X_test.to_csv(file_path + "X_test.csv", index=False)
y_train.to_csv(file_path + "y_train.csv", index=False)
y_test.to_csv(file_path + "y_test.csv", index=False)

