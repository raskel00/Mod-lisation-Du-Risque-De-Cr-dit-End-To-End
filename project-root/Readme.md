Voici la traduction cohérente en français de ton texte, en conservant intégralement le contenu, le message et les informations :

---

# **Documentation pour `utils.py`**

## **Vue d’ensemble**

Ce projet fournit un script utilitaire pour supporter un système de scoring de crédit construit avec un modèle d’apprentissage automatique. Le script inclut des fonctions pour prétraiter les données d’entrée, effectuer des prédictions et calculer des scores de crédit basés sur la probabilité de défaut. Les utilitaires sont conçus pour être modulaires, évolutifs et facilement intégrables dans une pipeline plus large d’évaluation du risque de crédit.

Le modèle prédictif est entraîné pour estimer la probabilité de défaut d’un prêt, et le score de crédit obtenu est conforme aux standards de l’industrie, allant de 300 (faible solvabilité) à 900 (excellente solvabilité). Ce script utilitaire joue un rôle crucial dans la préparation des données, la génération de prédictions et la fourniture d’informations exploitables.

---

## **Explication détaillée du code**

### 1. **Chargement du modèle**

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_data = joblib.load(r"project-root/model/model_data.pkl")
```

* **Objectif** : Charger le modèle sérialisé et les données associées (scaler, features, colonnes à standardiser) depuis un fichier `.pkl`.
* **Composants chargés** :

  * **`model`** : Le modèle d’apprentissage automatique entraîné (ex. XGBoost).
  * **`scaler`** : Un objet `StandardScaler` pour normaliser les variables numériques.
  * **`features`** : La liste des features utilisées pour la prédiction.
  * **`columns_to_scale`** : Les colonnes numériques à standardiser.

### 2. **Préparation des données**

```python
def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    data_input = {...}
    df = pd.DataFrame([data_input])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    df = df[features]
    return df
```

* **Objectif** : Préparer les données fournies par l’utilisateur pour la prédiction en :

  1. Collectant les entrées brutes dans un dictionnaire.
  2. Transformant ce dictionnaire en un DataFrame Pandas.
  3. Standardisant les colonnes spécifiées avec le `scaler` chargé.
  4. Sélectionnant uniquement les features utilisées par le modèle.

* **Calculs clés** :

  * Ratio prêt/revenu (`lti`) : calculé pour capturer la capacité de remboursement. Si le revenu est nul, la valeur par défaut est 0 pour éviter les divisions par zéro.
  * Encodage one-hot pour les variables catégorielles comme `loan_purpose`, `loan_type` et `residence_type`.

### 3. **Calcul du score de crédit**

```python
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    default_probability = model.predict_proba(input_df)[:, 1]
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length
    ...
    return default_probability.flatten()[0], int(credit_score), rating
```

* **Objectif** : Calculer le score de crédit et attribuer une note de crédit basée sur les prédictions du modèle.
* **Étapes** :

  1. Prédire la **probabilité de défaut** avec le modèle.
  2. Calculer la **probabilité de non-défaut** (complément de la probabilité de défaut).
  3. Déduire le **score de crédit** via une transformation linéaire sur une échelle de 300 à 900.
  4. Déterminer la **note de crédit** :

     * Poor (Faible) : 300–499
     * Average (Moyenne) : 500–649
     * Good (Bonne) : 650–749
     * Excellent : 750–900

### 4. **Fonction de prédiction**

```python
def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
            loan_amount, loan_tenure_months, total_loan_months, 
            loan_purpose, loan_type, residence_type):
    input_df = data_preparation(...)
    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating
```

* **Objectif** : Combiner préparation des données et calcul du score de crédit dans une fonction unique pour simplifier les prédictions.
* **Entrées** :

  * Données fournies par l’utilisateur, numériques (ex. `age`, `income`) et catégorielles (ex. `loan_purpose`).
* **Sorties** :

  * **Probabilité de défaut** : risque que l’utilisateur fasse défaut.
  * **Score de crédit** : valeur numérique représentant la solvabilité.
  * **Note** : étiquette descriptive (Poor, Average, Good, Excellent).

---

## **Explication conceptuelle**

### **Système de scoring de crédit**

Le script utilitaire est un composant clé d’un système de scoring de crédit basé sur l’apprentissage automatique. Le scoring de crédit évalue le risque de défaut d’un emprunteur, aidant les institutions financières dans les décisions d’octroi de prêts. Cette implémentation utilise un scoring basé sur les probabilités pour générer un score allant de 300 à 900, comparable aux standards de l’industrie.

### **Transformation des données**

* Le prétraitement garantit que les entrées brutes sont standardisées et correspondent au format attendu par le modèle.
* Les colonnes numériques sont mises à l’échelle via `StandardScaler`, améliorant la stabilité et la performance du modèle.

### **Prédiction du modèle**

* Le modèle pré-entraîné prédit la probabilité de défaut.
* Le score de crédit est calculé en utilisant une base et une échelle, traduisant les probabilités de défaut en une note compréhensible.

### **Interprétabilité**

* Le score calculé et la note attribuée fournissent des informations interprétables sur la solvabilité.
* Le système relie le score à des probabilités mesurables, assurant transparence et fiabilité.

---

## **Fonctionnalités clés**

1. **Design modulaire** : les fonctions sont autonomes et réutilisables.
2. **Évolutivité** : supporte différents formats d’entrée et peut intégrer de nouvelles features ou modèles.
3. **Conformité** : les scores respectent les normes industrielles, facilitant l’adoption.

---

## **Mode d’utilisation**

1. **Configurer l’environnement** :

   * Installer les dépendances (`joblib`, `numpy`, `pandas`, `scikit-learn`).
   * Charger le modèle sérialisé avec `joblib.load()`.

2. **Préparer les données d’entrée** :

   * Fournir les données nécessaires (âge, revenu, détails du prêt…) à la fonction `predict`.

3. **Effectuer les prédictions** :

   * Appeler `predict` pour obtenir la probabilité de défaut, le score de crédit et la note.

4. **Intégration** :

   * Utiliser le score et la note pour prendre des décisions dans les flux financiers.

---

# **Documentation pour `main.py`**

## **Vue d’ensemble**

Le fichier `main.py` sert d’interface frontend pour un système de Modélisation Du Risque De Crédit. Construit avec Streamlit, cette application permet aux utilisateurs de saisir les informations d’un emprunteur et de calculer la probabilité de défaut, le score de crédit et la note de risque. L’application fournit à la fois des informations intuitives et des résultats exploitables, constituant un outil pratique pour les institutions financières.

---

## **Explication conceptuelle**

### **Modélisation Du Risque De Crédit**

La Modélisation Du Risque De Crédit évalue la probabilité qu’un emprunteur fasse défaut sur un prêt. L’application utilise un modèle d’apprentissage automatique pour évaluer le risque en fonction des caractéristiques de l’emprunteur et du prêt. Les sorties incluent :

* **Probabilité de défaut** : exprimée en pourcentage.
* **Score de crédit** : valeur numérique (300–900).
* **Note de crédit** : évaluation qualitative (Poor, Average, Good, Excellent).

### **Fonctionnalités de l’application**

* **Entrées interactives** : ajustement dynamique des paramètres de l’emprunteur et du prêt.
* **Évaluation du risque en temps réel** : calcul instantané de la probabilité de défaut, du score et de la note.
* **Interface conviviale** : Streamlit offre un design propre et réactif.

---

## **Explication détaillée du code**

### 1. **Configuration de la page**

```python
st.set_page_config(page_title="Jeff Finance: Credit Risk Modelling", page_icon="📊", layout="centered")
st.title("📊 Jeff Finance: Credit Risk Modelling")
```

* **Objectif** : Configurer le titre, l’icône et la mise en page de l’app. Crée une interface accueillante.

### 2. **Instructions dans la barre latérale**

```python
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Remplir les champs nécessaires à droite.
    2. Ajuster les curseurs et menus déroulants pour les entrées interactives.
    3. Cliquer sur 'Calculate Risk' pour afficher les résultats.
    """)
    st.image("project-root/Jeff Finance.JPG", caption="Your Trusted Finance Partner")
```

* **Objectif** : Fournir des instructions claires pour faciliter l’utilisation.
* **Image intégrée** : Ajoute un logo ou image pertinente pour l’identité visuelle.

### 3. **Champs d’entrée**

#### Informations sur l’emprunteur

```python
col1, col2, col3 = st.columns(3)
age = col1.number_input("📅 Age", min_value=18, max_value=100, value=28, help="Entrez votre âge (18-100).")
income = col2.number_input("💰 Revenu (Annuel)", min_value=0, max_value=5000000, value=290875, step=50000, help="Votre revenu annuel.")
loan_amount = col3.number_input("🏦 Montant du prêt", min_value=0, value=2560000, help="Montant total du prêt souhaité.")
```

* **Objectif** : Collecter les informations démographiques et financières principales.

#### Indicateurs du prêt

```python
lti = loan_amount / income if income > 0 else 0
st.metric(label="Loan-to-Income Ratio (LTI)", value=f"{lti:.2f}", help="Ratio du montant du prêt par rapport au revenu.")
```

* **Objectif** : Calculer le ratio prêt/revenu (LTI) pour évaluer l’endettement.

#### Détails du prêt

```python
loan_tenure_months = col4.slider("⏳ Durée du prêt (mois)", min_value=6, max_value=240, step=6, value=36, help="Durée du prêt en mois.")
avg_dpd_per_dm = col5.number_input("⚠ Moy. DPD", min_value=0, value=0, help="Jours de défaut moyen par mois, 0 si pas d’historique.")
dmtlm = col6.slider("📅 DMTLM (Ratio mois délinquants / prêt)", min_value=0, max_value=100, value=0, help="Ratio de délinquance, 0 si pas de prêts.")
```

* **Objectif** : Collecter les détails spécifiques au prêt.

#### Objet du prêt et autres détails

```python
credit_utilization_ratio = col7.slider("💳 Utilisation du crédit (%)", min_value=0, max_value=100, value=0, help="Pourcentage de crédit utilisé, 0 si aucun.")
total_loan_months = col8.number_input("📜 Total mois de prêt", min_value=0, value=0, help="Durée cumulée des prêts, 0 si aucun.")
loan_purpose = col9.selectbox("🎯 Objet du prêt", ['Education', 'Home', 'Auto', 'Personal'], help="Objet du prêt.")
```

* **Objectif** : Capturer l’utilisation du crédit, la durée cumulée et l’objet du prêt.

#### Type de prêt et résidence

```python
loan_type = col10.radio("🔑 Type de prêt", ['Unsecured', 'Secured'], help="Choisir le type de prêt.")
residence_type = col11.selectbox("🏡 Type de résidence", ['Owned', 'Rented', 'Mortgage'], help="Type de résidence actuel.")
```

* **Objectif** : Identifier le type de prêt et la situation résidentielle.

### 4. **Calcul du risque**

```python
if st.button("Calculate Risk"):
    probability, credit_score, rating = predict(...)
    st.success("✅ Évaluation du risque terminée !")
    st.write(f"**Probabilité de défaut :** {probability:.2%}")
    st.write(f"**Score de crédit :** {credit_score}")
    st.write(f"**Note :** {rating}")
```

* **Objectif** : Déclencher la prédiction lors du clic sur “Calculate Risk”.
* **Sorties** : Probabilité de défaut, score de crédit, note descriptive.

### 5. **Analyse du risque**

```python
if rating in ['Poor', 'Average']:
    st.warning("⚠ Profil à risque élevé. Envisagez d’améliorer les habitudes de crédit.")
else:
    st.info("🌟 Profil à faible risque. Approbation probable du prêt.")
```

* **Objectif** : Fournir des retours exploitables selon la note de crédit.

---

## **Fonctionnalités clés**

* **Design centré utilisateur** : Simplifie la modélisation du risque pour les non-techniciens.
* **Widgets interactifs** : Entrées dynamiques et résultats instantanés.
* **Analyse du risque** : Guide les décisions avec des informations claires.

---

## **Mode d’utilisation**

1. **Lancer l’application** :

   * Installer Streamlit et les dépendances.
   * Exécuter `streamlit run main.py`.

2. **Interagir avec l’interface** :

   * Saisir les détails de l’emprunteur, ajuster les paramètres et sélectionner les caractéristiques du prêt.
   * Cliquer sur “Calculate Risk” pour voir les résultats.

3. **Analyser et intégrer** :

   * Utiliser les résultats pour évaluer le profil de risque et prendre des décisions éclairées.

---

# **Documentation des hyperparamètres optimisés**

## **Vue d’ensemble**

Le modèle XGBoost utilisé dans ce projet a été optimisé avec **Optuna**, un framework avancé d’optimisation d’hyperparamètres. Ces hyperparamètres améliorent la performance du modèle en équilibrant précision, efficacité computationnelle et généralisation. Ci-dessous l’explication des hyperparamètres sélectionnés et leur importance.

---

## **Explication des hyperparamètres**

1. **`eta` (Taux d’apprentissage)** : `0.03962150782811734`

   * **Définition** : Contrôle la taille des pas lors de l’optimisation.
   * **Effet** : Un petit `eta` permet un apprentissage progressif, limitant l’overfitting. La valeur 0.0396 est conservatrice, idéale pour le fine-tuning.

2. **`max_depth`** : `3`

   * **Définition** : Profondeur maximale des arbres de décision.
   * **Effet** : Limite la complexité pour éviter l’overfitting. Une profondeur de 3 favorise la généralisation.

3. **`subsample`** : `0.6272358596011762`

   * **Définition** : Fraction des échantillons utilisés pour entraîner chaque arbre.
   * **Effet** : Prévient l’overfitting et introduit de la diversité (62,7 % des données).

4. **`colsample_bytree`** : `0.7136867658100697`

   * **Définition** : Fraction des features considérées pour chaque arbre.
   * **Effet** : Utilise 71,4 % des features pour réduire le risque d’overfitting tout en conservant la puissance prédictive.

5. **`n_estimators`** : `388`

   * **Définition** : Nombre d’arbres dans le modèle.
   * **Effet** : Permet un nombre suffisant d’itérations pour atteindre une haute précision sans surcoût computationnel.

---

## **Importance de ces hyperparamètres**

Ils équilibrent :

* **Performance** : Optimisation de métriques (AUC, Gini, KS).
* **Efficacité** : Limite la complexité et la charge computationnelle.
* **Généralisation** : Meilleure adaptation aux données non vues.

---

## **Framework d’optimisation**

Les hyperparamètres ont été ajustés via **Optuna**, utilisant :

* **Optimisation bayésienne** : exploration efficace de l’espace des hyperparamètres.
* **Fonction objectif** : maximisation de métriques comme AUC et Gini.
* **Critères d’arrêt** : arrêt automatique si pas d’amélioration significative.

---

## **Avantages du fine-tuning**

1. **Meilleure capacité prédictive** : Distinction accrue entre défaut et non-défaut.
2. **Réduction de l’overfitting** : `subsample` et `colsample_bytree` améliorent la généralisation.
3. **Entraînement efficace** : Minimisation du calcul inutile pour un déploiement plus pratique.

---

## **Application des hyperparamètres**

Pour reproduire ou adapter le modèle :

1. Utiliser ce dictionnaire dans la fonction XGBoost :

```python
params = {
   'eta': 0.03962150782811734,
   'max_depth': 3,
   'subsample': 0.6272358596011762,
   'colsample_bytree': 0.7136867658100697,
   'n_estimators': 388
}
```

2. Initialiser le classificate


ur XGBoost :

```python
from xgboost import XGBClassifier
model = XGBClassifier(**params)
```

3. Entraîner le modèle sur vos données :

```python
model.fit(X_train, y_train)
```
