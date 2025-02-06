
# Script d'évaluation du modèle:
# L’objectif de ce script est d’évaluer les performances du modèle sur les données de test et de générer des prédictions

# Import librairies
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

# Chargement des données de test et du modèle
file_path = "../../data/processed_data/"
X_test_scaled = pd.read_csv(file_path + "X_test_scaled.csv")
y_test = pd.read_csv(file_path + "y_test.csv")

file_path = "../../models/"
with open(file_path + "random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prédictions
y_pred = model.predict(X_test_scaled)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des résultats
metrics = {"mse": mse, "r2": r2}
file_path = "../../metrics/"
with open(file_path + "scores.json", "w") as f:
    json.dump(metrics, f)

# Sauvegarder des prédictions
file_path = "../../data/"
predictions = pd.DataFrame({"y_test": y_test.values.flatten(), "y_pred": y_pred})
predictions.to_csv(file_path + "predictions.csv", index=False)

