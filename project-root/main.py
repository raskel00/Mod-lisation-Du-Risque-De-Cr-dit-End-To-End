# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:29:38 2025

@author: Admin
"""

# import os
# print(os.getcwd())

import streamlit as st
from utils import predict
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "assets", "jeff_finance.jpg")


# Définir la configuration de la page et le titre
st.set_page_config(page_title="Modélisation Du Risque De Crédit", page_icon="📊", layout="centered")
st.title("📊 Modélisation Du Risque De Crédit")

# Barre latérale – Instructions pour l’utilisateur
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Remplissez les champs nécessaires à droite.
    2. Ajustez les curseurs et les menus déroulants pour des entrées interactives.
    3. Cliquez sur 'Calculate Risk' pour afficher les résultats.
    """)
    st.image(IMG_PATH, caption="Votre partenaire financier de confiance")  # Ajouter une image ou un logo pertinent.

# Champs d’entrée
st.subheader("💼 Informations Client")

# Ligne 1 : Âge, Revenu, Montant du prêt
col1, col2, col3 = st.columns(3)

age = col1.number_input("Age", min_value=18, max_value=100, value=28, help="Entrez votre âge (18-100).")
income = col2.number_input("Revenu (Annuel)", min_value=0, max_value=5000000, value=290875, step=50000, help="Votre revenu annuel en unités monétaires.")
loan_amount = col3.number_input("Montant du Prêt", min_value=0, value=2560000, help="Montant total que vous souhaitez emprunter.")

# Ligne 2 : Indicateurs du prêt
st.subheader("📊 Indicateurs du Prêt")
lti = loan_amount / income if income > 0 else 0
st.metric(label="Ratio Prêt/Revenu (LTI)", value=f"{lti:.2f}", help="Montre le ratio entre le montant du prêt et votre revenu.")

# Ligne 3 : Durée du prêt, Avg DPD, DMTLM
st.subheader("📑 Détails du Prêt")
col4, col5, col6 = st.columns(3)

loan_tenure_months = col4.slider("Durée du Prêt (Mois)", min_value=6, max_value=240, step=6, value=36, help="Sélectionnez la durée du prêt en mois.")
avg_dpd_per_dm = col5.number_input("Moyenne des Jours de Retard (DPD) par Mois en défaut", min_value=0, value=0, help="Moyenne des jours de retard (Defaults), mettre 0 en absence d’historique de prêt.")
dmtlm = col6.slider("DMTLM (Ratio Mois Délinquants / Mois de Prêt)", min_value=0, max_value=100, value=0, help="Ratio de délinquance, 0 si aucun prêt.")

# Ligne 4 : Utilisation du crédit, Mois totaux de prêt, Objet du prêt
st.subheader("🏡 Objet du Prêt")
col7, col8, col9 = st.columns(3)

credit_utilization_ratio = col7.slider("Utilisation du Crédit (%)", min_value=0, max_value=100, value=0, help="Pourcentage de crédit utilisé, 0 si aucun crédit.")
total_loan_months = col8.number_input("Mois Totaux de Prêt", min_value=0, value=0, help="Durée cumulée de tous les prêts, 0 si aucun prêt.")
loan_purpose = col9.selectbox("Objet du Prêt", ['Éducation', 'Maison', 'Auto', 'Personnel'], help="Objet du prêt.")

# Ligne 5 : Type de prêt, Type de résidence
st.subheader("🏠 Type de Prêt et Résidence")
col10, col11 = st.columns(2)

loan_type = col10.radio("Type de Prêt", ['Non Garanti', 'Garanti'], help="Choisissez le type de prêt.")
residence_type = col11.selectbox("Type de Résidence", ['Propriétaire', 'Loué', 'Hypothèque'], help="Votre type de résidence actuel.")

# Bouton d'action
if st.button("Calculate Risk"):
    # Appeler la fonction `predict` avec les champs saisis
    probability, credit_score, rating = predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                                                loan_amount, loan_tenure_months, total_loan_months,
                                                loan_purpose, loan_type, residence_type)

    # Affichage des résultats
    st.success("✅ Évaluation du Risque Terminée !")
    st.write(f"**Probabilité de Défaut :** {probability:.2%}")
    st.write(f"**Score de Crédit :** {credit_score}")
    st.write(f"**Notation :** {rating}")

    # Insights sur le risque
    if rating in ['Poor', 'Average']:
        st.warning("⚠ L’emprunteur présente un profil à haut risque. Envisagez d'améliorer les habitudes de crédit.")
    else:
        st.info("🌟 L’emprunteur présente un profil à faible risque. L'approbation du prêt est probable.")

