
# Script d'entraînement du modèle:
# L’objectif de ce script est d’utiliser les meilleurs paramètres pour entraîner le modèle choisi

# Import librairies
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Chargement des données
file_path = "../../data/processed_data/"
X_train_scaled = pd.read_csv(file_path + "X_train_scaled.csv")
y_train = pd.read_csv(file_path + "y_train.csv")

# Chargement des meilleurs paramètres
file_path = "../../models/"
with open(file_path + "best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Entraînement du modèle avec les meilleurs paramètres
model = RandomForestRegressor(n_estimators=best_params["n_estimators"],
                              max_depth=best_params["max_depth"])
model.fit(X_train_scaled, y_train.to_numpy().ravel())

# Sauvegarde du modèle entraîné
with open(file_path + "random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

