
# Script de GridSearch pour la recherche des meilleurs paramètres:
# L’objectif de ce script est de tester plusieurs modèles de régression et de trouver les meilleurs paramètres à utiliser

# Import librairies
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Chargement des données
file_path = "../../data/processed_data/"
X_train_scaled = pd.read_csv(file_path + "X_train_scaled.csv")
y_train = pd.read_csv(file_path + "y_train.csv")

# Modèles à tester
rfr = RandomForestRegressor()
# Paramètres à tester
param_grid = {
    "n_estimators": [100, 200], "max_depth": [10, 20]
}

# GridSearch pour chaque modèle
grid_search = GridSearchCV(rfr, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train.to_numpy().ravel())
best_params = grid_search.best_params_

# Sauvegarde des meilleurs paramètres
file_path = "../../models/"
with open(file_path + "best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)
