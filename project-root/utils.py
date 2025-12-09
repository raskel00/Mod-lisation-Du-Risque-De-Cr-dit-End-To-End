# -*- coding: utf-8 -*-
"""
Créé le Lun 09 Déc 2025 à 21:15:47

@author: Admin
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_data.pkl")

model_data = joblib.load(MODEL_PATH)


# =============================================================================
# import importlib
#
# # Liste des bibliothèques à vérifier
# libraries = [
#     "joblib",
#     "pandas",
#     "numpy",
#     "streamlit",
#     "sklearn",
#     "xgboost"
# ]
#
# # Boucle pour afficher la version de chaque bibliothèque
# for library in libraries:
#     try:
#         module = importlib.import_module(library)
#         print(f"{library} version: {module.__version__}")
#     except ImportError:
#         print(f"{library} n'est pas installé.")
#     except AttributeError:
#         print(f"Impossible de déterminer la version pour {library}.")
# =============================================================================

model = model_data['model']
print(model)

scaler = model_data['scaler']
print(scaler)

features = model_data['features']
print(features)
len(features)

columns_to_scale = model_data['cols_to_scale']
print(columns_to_scale)


def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    """
    Prépare les données d'entrée pour le modèle :
    - encodage des variables catégorielles
    - calcul du ratio LTI
    - normalisation des colonnes nécessaires
    """
    data_input = {'age': age,
                  'avg_dpd_per_dm': avg_dpd_per_dm,
                  'credit_utilization_ratio': credit_utilization_ratio,
                  'dmtlm': dmtlm,
                  'income': income,
                  'loan_amount': loan_amount,
                  'lti': loan_amount / income if income > 0 else 0,
                  'total_loan_months': total_loan_months,
                  'loan_tenure_months': loan_tenure_months,
                  'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
                  'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
                  'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
                  'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
                  'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
                  'residence_type_Rented': 1 if residence_type == 'Rented' else 0}
    
    df = pd.DataFrame([data_input])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    df = df[features]
    
    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """
    Calcule le score de crédit à partir de la probabilité de défaut :
    - utilise la probabilité de non-défaut pour générer un score sur une échelle 300–900
    - attribue une catégorie ('Poor', 'Average', 'Good', 'Excellent')
    """
    default_probability = model.predict_proba(input_df)[:, 1]  # Probabilité de défaut
    non_default_probability = 1 - default_probability

    # Calcul du score de crédit basé sur les probabilités
    credit_score = base_score + non_default_probability.flatten() * scale_length
    
    # Déterminer la catégorie de notation en fonction du score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # en cas de score inattendu

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score), rating


def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    """
    Fonction principale de prédiction :
    - prépare les données
    - calcule la probabilité de défaut, le score de crédit et la notation
    """
    input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                         loan_amount, loan_tenure_months, total_loan_months, 
                         loan_purpose, loan_type, residence_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


