# Ingénierie des Features et Construction de Modèle

## Vue d’ensemble

Ce projet démontre un pipeline complet pour l’ingénierie des features et la préparation d’un modèle, ciblant spécifiquement une tâche de prédiction de défaut de prêt. Le processus inclut le chargement des données, l’ingénierie des features, la gestion de la multicolinéarité, la sélection des features les plus pertinentes sur la base de la **valeur d’information (IV)**, l’encodage des variables catégorielles, la normalisation des features numériques et la préparation du dataset pour les modèles de machine learning.

Le projet est structuré pour mettre en avant les étapes clés de prétraitement des données et de sélection des features, constituant ainsi un excellent exemple pour construire une base solide pour la modélisation prédictive. Ci-dessous, une explication détaillée des étapes implémentées et du code Python.

---

## **1. Chargement des librairies et des données nécessaires**

Le projet commence par l’importation des librairies essentielles, notamment :

* **Pandas** pour la manipulation des données.
* **Numpy** pour les opérations numériques.
* **Seaborn** et **Matplotlib** pour la visualisation des données.
* **Joblib** pour sauvegarder et charger les modèles efficacement.
* **Warnings** pour supprimer les avertissements inutiles lors de l’exécution.

Les données sont chargées depuis un fichier CSV (`cleaned_data.csv`), et les premières étapes incluent la suppression des colonnes redondantes (`Unnamed: 0`) et l’examen de la structure et des types des données.

```python
df = pd.read_csv('cleaned_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
```

---

## **2. Ingénierie des Features (Feature Engineering)**

### 2.1 Ratio Prêt/Revenu (LTI)

Le ratio Prêt/Revenu (LTI) est une feature clé en analyse financière. Il est calculé comme le ratio du montant du prêt sur le revenu et arrondi à deux décimales.

```python
df['lti'] = np.round(df['loan_amount'] / df['income'], 2)
```

Visualisations :

* **KDE Plot** : pour analyser la distribution du LTI selon le statut de défaut.
* **Histogramme** : pour explorer la distribution du LTI et sa relation avec le taux de défaut.

---

### 2.2 Ratio Mois en défaut / Mois de Prêt (DMTLM)

Cette feature mesure le pourcentage de mois où le prêt a été en défaut par rapport à sa durée totale, fournissant des informations sur le comportement de remboursement de l’emprunteur.

```python
df['dmtlm'] = np.round((df['delinquent_months'] / df['total_loan_months']) * 100, 1)
```

Visualisations :

* Des graphiques KDE et histogrammes similaires sont utilisés pour étudier ce ratio de délinquance.

---

### 2.3 Moyenne des Jours de Retard (DPD) par Mois en défaut

Cette feature calcule le nombre moyen de jours de retard par mois en défaut, quantifiant la gravité des retards.

```python
df['avg_dpd_per_dm'] = np.where(
    df['delinquent_months'] > 0,
    np.round(df['total_dpd'] / df['delinquent_months'], 1),
    0
)
```

---

## **3. Suppression des Features Non Pertinentes**

Les features non pertinentes comme les identifiants clients, les détails de localisation et les dates sont supprimés pour réduire le bruit et éviter le surapprentissage.

```python
df = df.drop(['cust_id', 'city', 'state', 'zipcode', 'disbursal_date', 'installment_start_dt'], axis=1)
```

---

## **4. Gestion de la Multicolinéarité**

### Facteur d’inflation de la variance (VIF)

Le VIF est utilisé pour identifier la multicolinéarité entre les features numériques. Les features avec des valeurs VIF élevées sont supprimées de manière itérative.

```python
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data
```

Après calcul du VIF sur les données normalisées et non normalisées, plusieurs features sont retirées pour gérer la multicolinéarité.

---

## **5. Sélection des Features avec la Valeur d’Information (IV)**

L’IV quantifie le pouvoir prédictif de chaque feature, ce qui en fait une métrique robuste pour la sélection des features. Les features avec IV < 0.02 sont exclues.

```python
def calculate_iv_for_train(X_train, y_train):
    ...
iv_data = calculate_iv_for_train(X_train, y_train)
selected_features = iv_data[iv_data['IV'] >= 0.02]['Feature'].tolist()
```

Un graphique en barres est tracé pour visualiser l’importance des features selon l’IV.

---

## **6. Encodage des Variables Catégorielles et Normalisation des Features Numériques**

### Encodage

Les features catégorielles sont encodées en one-hot pour les rendre adaptées aux modèles de machine learning. Les catégories manquantes dans le jeu de test sont gérées par alignement des colonnes.

```python
X_train_encoded = pd.get_dummies(X_train_selected, columns=cat_cols, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
```

### Normalisation

Les features numériques sont standardisées avec `StandardScaler` pour de meilleures performances des modèles.

```python
scaler = StandardScaler()
X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])
```

---

## **7. Sauvegarde des Détails de Préparation du Modèle**

Le dataset final et les détails de normalisation sont sauvegardés pour le déploiement du modèle. Cela garantit la cohérence entre l’entraînement et la prédiction.

```python
model_data = {
    'model': None,
    'scaler': scaler,
    'features': X_train_encoded.columns.tolist(),
    'cols_to_scale': num_cols,
}
```

---

## **8. Construction et Évaluation du Modèle**

La première section configure les imports nécessaires et définit une fonction d’aide pour l’entraînement et l’évaluation des modèles.

#### Composants Clés :

1. **Imports pour les métriques** :

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

* Utilisés pour calculer et visualiser la performance du modèle.

2. **Fonction d’Aide** :

```python
def build_evaluate_model(model, model_name, train_x, train_y, test_x, test_y):
```

* Entraîne un modèle donné et évalue sa performance.
* **Étapes de la fonction** :

  * Ajuster le modèle aux données d’entraînement.
  * Calculer et afficher le score d’entraînement.
  * Prédire sur les données de test et calculer :

    * **Accuracy Score**
    * **Classification Report** (précision, rappel, F1-score, support)
    * **Confusion Matrix** (visualisée via heatmap avec `sns.heatmap()`).

---

### **Modèles de Référence**

Trois modèles de base sont construits et évalués avec la fonction d’aide.

1. **Régression Logistique** :

```python
lr_model = build_evaluate_model(
    model=LogisticRegression(), 
    model_name='Logistic Regression', 
    train_x=X_train_encoded, 
    train_y=y_train,
    test_x=X_test_encoded, 
    test_y=y_test
)
```

2. **Random Forest** :

```python
rf_model = build_evaluate_model(
    model=RandomForestClassifier(), 
    model_name='Random Forest', 
    train_x=X_train_encoded, 
    train_y=y_train,
    test_x=X_test_encoded, 
    test_y=y_test
)
```

3. **Extreme Gradient Boosting (XGBoost)** :

```python
xgb_model = build_evaluate_model(
    model=XGBClassifier(), 
    model_name='Extreme Gradient Boost', 
    train_x=X_train_encoded, 
    train_y=y_train,
    test_x=X_test_encoded, 
    test_y=y_test
)
```
---

## Optimisation des hyperparamètres avec Randomized Search CV

Cette section optimise les hyperparamètres des modèles à l’aide de **RandomizedSearchCV**.

### Éléments clés :

#### Grilles d’hyperparamètres :

Définition de l’espace de recherche pour chaque modèle :

* **Régression Logistique :**
  Variation de *penalty* (régularisation), *C* (intensité de la régularisation), *solver*, et *max_iter*.

* **Random Forest :**
  Expérimentation de *n_estimators* (nombre d’arbres), *max_depth*, *min_samples_split*, etc.

* **XGBoost :**
  Ajustement de paramètres tels que *n_estimators*, *learning_rate*, *max_depth*, etc.

#### Boucle RandomizedSearchCV :

```python
for model_name, (model, param_grid) in models.items():
```

Pour chaque modèle :

* Initialise un objet **RandomizedSearchCV**.
* Effectue une recherche aléatoire sur l’espace d’hyperparamètres défini.
* Affiche et enregistre les meilleurs paramètres.
* Sauvegarde le meilleur modèle avec **joblib**.

#### Évaluation du modèle :

Après l’optimisation, chaque meilleur modèle est chargé et évalué sur les données de test.

---

## Optimisation des hyperparamètres avec Optuna

**Optuna** est un framework d’optimisation utilisé pour la recherche d’hyperparamètres.

### Éléments clés :

#### Fonctions objectifs :

Chaque fonction définit l’espace de recherche pour un modèle spécifique.

##### Régression Logistique :

```python
def objective_logreg(trial):
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"]) if solver != "saga" else "l2"
```

Ajuste la régularisation (*C*) et les paramètres du solveur.

##### Random Forest :

```python
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
```

Ajuste le nombre d’arbres, la profondeur et les critères de division.

##### XGBoost :

```python
def objective_xgb(trial):
    eta = trial.suggest_loguniform("eta", 0.01, 0.5)
    max_depth = trial.suggest_int("max_depth", 3, 10)
```

Se concentre sur des paramètres spécifiques au boosting comme le *learning rate* (*eta*).

#### Optimisation :

```python
study_logreg.optimize(objective_logreg, n_trials=100)
```

Chaque modèle est optimisé sur un nombre fixe d’essais.
Les meilleurs paramètres et scores de précision sont affichés.

---

## Évaluation des meilleurs modèles

Les meilleurs modèles (issus de RandomizedSearchCV et Optuna) sont entraînés et évalués sur les données de test.

### Étapes :

* Entraîner le modèle avec les meilleurs hyperparamètres.
* Prédire sur l’ensemble de test.
* Calculer :

  * l’Accuracy
  * le Classification Report
  * la Matrice de Confusion (visualisée en heatmap)

**Exemple (Régression Logistique) :**

```python
best_logreg = LogisticRegression(**study_logreg.best_params, random_state=42, max_iter=1000)
best_logreg.fit(X_train_encoded, y_train)
```

---

## Undersampling pour gérer le déséquilibre des classes

```python
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Appliquer l'undersampling
rus = RandomUnderSampler(random_state=42)
X_train_balanced_under, y_train_balanced_under = rus.fit_resample(X_train_encoded, y_train)

# Afficher la distribution des classes
print("Distribution des classes avant undersampling :", Counter(y_train))
print("Distribution des classes après undersampling :", Counter(y_train_balanced_under))
```

### Explication :

* **RandomUnderSampler** réduit la classe majoritaire pour équilibrer le dataset.
* Affiche les distributions des classes avant et après rééchantillonnage.

---

## Modèles de base sur données undersampled

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Modèles de base
models = {
    'Régression Logistique': LogisticRegression(random_state=42),
    'Forêt Aléatoire': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Évaluer les modèles
for name, model in models.items():
    print(f"Évaluation de {name} sur les données sous-échantillonnées...")
    build_evaluate_model(model, X_train_balanced_under, y_train_balanced_under, X_test_encoded, y_test)

```

### Explication :

* Les modèles de base sont entraînés sur le jeu de données sous-échantillonné et évalués à l’aide d’une fonction utilitaire (*build_evaluate_model*).
  
* Affiche les métriques de performance (accuracy, rapport de classification, etc.).

---
# **Recherche Aléatoire (Randomized Search CV) sur les Données Sous-Échantillonnées**

```python
from sklearn.model_selection import RandomizedSearchCV

# Grilles d'hyperparamètres
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]},
    'XGBoost': {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]}
}

# Randomized search
for name, model in models.items():
    print(f"Running RandomizedSearchCV for {name}...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, n_iter=10, random_state=42)
    random_search.fit(X_train_balanced_under, y_train_balanced_under)
    print(f"Best params for {name}:", random_search.best_params_)
    build_evaluate_model(random_search.best_estimator_, X_train_balanced_under, y_train_balanced_under, X_test_encoded, y_test)
```

**Explication :**

* Effectue l’optimisation d’hyperparamètres avec **RandomizedSearchCV**.
* Évalue les meilleurs modèles à l’aide de **build_evaluate_model**.

---

# **Optuna sur les Données Sous-Échantillonnées**

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective_logistic(trial):
    C = trial.suggest_loguniform('C', 0.01, 10)
    model = LogisticRegression(C=C, random_state=42)
    score = cross_val_score(model, X_train_balanced_under, y_train_balanced_under, cv=3).mean()
    return score

study_logistic = optuna.create_study(direction='maximize')
study_logistic.optimize(objective_logistic, n_trials=20)

print("Best Logistic Regression params:", study_logistic.best_params_)
```

**Explication :**

* Optuna est utilisé pour la recherche d’hyperparamètres via une fonction objectif.
* Le score de validation croisée est optimisé, et les meilleurs paramètres sont affichés.

---

# **Suréchantillonnage pour Gérer le Déséquilibre de Classe**

```python
from imblearn.over_sampling import SMOTE

# Apply oversampling
smote = SMOTE(random_state=42)
X_train_balanced_over, y_train_balanced_over = smote.fit_resample(X_train_encoded, y_train)

# Print class distribution
print("Class distribution before oversampling:", Counter(y_train))
print("Class distribution after oversampling:", Counter(y_train_balanced_over))
```

**Explication :**

* SMOTE génère des échantillons synthétiques pour la classe minoritaire afin d’équilibrer les données.
* Affiche les distributions avant et après.

---

# **Modèles de Base sur les Données Suréchantillonnées**

```python
# Evaluate models on oversampled data
for name, model in models.items():
    print(f"Evaluating {name} on oversampled data...")
    build_evaluate_model(model, X_train_balanced_over, y_train_balanced_over, X_test_encoded, y_test)
```

**Explication :**

* Entraîne et évalue les mêmes modèles baselines (Logistic Regression, Random Forest, XGBoost) sur les données suréchantillonnées.

---

# **Randomized Search CV sur les Données Suréchantillonnées**

```python
# Randomized search for oversampled data
for name, model in models.items():
    print(f"Running RandomizedSearchCV for {name} on oversampled data...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, n_iter=10, random_state=42)
    random_search.fit(X_train_balanced_over, y_train_balanced_over)
    print(f"Best params for {name}:", random_search.best_params_)
    build_evaluate_model(random_search.best_estimator_, X_train_balanced_over, y_train_balanced_over, X_test_encoded, y_test)
```

**Explication :**

* Comme pour l’undersampling, RandomizedSearchCV ajuste les hyperparamètres sur les données suréchantillonnées et évalue les performances.

---

# **Optuna sur les Données Suréchantillonnées**

```python
def objective_xgboost(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    score = cross_val_score(model, X_train_balanced_over, y_train_balanced_over, cv=3).mean()
    return score

study_xgboost = optuna.create_study(direction='maximize')
study_xgboost.optimize(objective_xgboost, n_trials=20)

print("Best XGBoost params:", study_xgboost.best_params_)
```

**Explication :**

* Optuna est utilisé ici pour optimiser XGBoost sur les données suréchantillonnées.
* Retourne et affiche les meilleurs hyperparamètres.

---

# **Métriques d'Évaluation**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example evaluation
y_pred = model.predict(X_test_encoded)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

**Explication :**

* Mesure les performances via accuracy, précision, recall, F1-score.
* La matrice de confusion est visualisée pour une meilleure interprétation.

---

# **Comparaison des Meilleurs Modèles : Code + Explications**

### 1. **Charger les Modèles et Calculer les Métriques**

```python
from sklearn.metrics import precision_score, recall_score

model_files = [
    "Logistic Regression_best_model_over.pkl",
    "Logistic Regression_best_model_under.pkl",
    "XGBoost_best_model_under.pkl",
    "logreg_optuna_over.pkl",
    "logreg_optuna_under.pkl",
    "lr_model_over.pkl",
    "lr_model_under.pkl",
    "xgb_model_under.pkl",
    "xgb_optuna_under.pkl"
]

results = []

for model_file in model_files:
    model = joblib.load(model_file)

    y_pred = model.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    results.append({
        "Model": model_file,
        "Accuracy": accuracy,
        "Precision (Default)": precision,
        "Recall (Default)": recall
    })

results_df = pd.DataFrame(results)
results_df
```

**Explication :**

* Objectif : comparer les performances des modèles enregistrés.
* Pour chaque modèle :

  * Chargement
  * Prédictions
  * Calcul de accuracy, précision et recall
  * Stockage dans un DataFrame

---

### 2. **Mettre en Évidence les Meilleurs Modèles**

```python
results_df[(results_df['Precision (Default)'] == results_df['Precision (Default)'].max()) | 
           (results_df['Recall (Default)'] == results_df['Recall (Default)'].max())]
```

**Explication :**

* Sélectionne les modèles ayant la meilleure précision ou recall sur la classe "Default".

---

### 3. **Visualisation Comparée des Modèles**

```python
plt.figure(figsize=(20, 8))

results_melted = results_df.melt(id_vars="Model",
                                 value_vars=["Accuracy", "Precision (Default)", "Recall (Default)"],
                                 var_name="Metric", value_name="Score")

sns.barplot(x="Score", y="Model", hue="Metric", data=results_melted, palette="Set2")

plt.title("Model Comparison Based on Test Metrics", fontsize=16)
plt.xlabel("Score", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(title="Metric", loc="upper right", fontsize=12, bbox_to_anchor=(1.4, 1))
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.show()
```

**Explication :**

* Compare graphiquement accuracy, précision et recall pour chaque modèle.

---

### **4. Charger les modèles pour une évaluation plus approfondie**

```python
loaded_models = {model_file: joblib.load(model_file) for model_file in model_files}
```

**Explication :**

Charge tous les modèles dans un dictionnaire pour une utilisation ultérieure dans les évaluations détaillées.

---

### **5. Évaluer les modèles avec des statistiques par déciles**

```python
from sklearn.metrics import roc_curve, roc_auc_score

def evaluate_model(model, model_name, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"default_truth": y_test, "default_probability": y_prob})
    df = df.sort_values(by="default_probability", ascending=False).reset_index(drop=True)

    df["decile"] = pd.qcut(df["default_probability"], 10, labels=range(1, 11))

    decile_stats = df.groupby("decile").agg(
        min_probability=("default_probability", "min"),
        max_probability=("default_probability", "max"),
        event_count=("default_truth", "sum"),
        non_event_count=("default_truth", lambda x: (x == 0).sum())
    )

    decile_stats["event_rate"] = decile_stats["event_count"] / decile_stats["event_count"].sum()
    decile_stats["non_event_rate"] = decile_stats["non_event_count"] / decile_stats["non_event_count"].sum()
    decile_stats["cum_event_rate"] = decile_stats["event_rate"].cumsum()
    decile_stats["cum_non_event_rate"] = decile_stats["non_event_rate"].cumsum()

    decile_stats["ks"] = abs(decile_stats["cum_event_rate"] - decile_stats["cum_non_event_rate"]) * 100
    ks_stat = decile_stats["ks"].max()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    gini_coefficient = 2 * auc_score - 1

    print(f"\n=== Model: {model_name} ===")
    print(f"KS Statistic: {ks_stat:.2f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"Gini Coefficient: {gini_coefficient:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Random Model")
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return decile_stats
```

**Explication :**

**Objectif :** Réaliser une évaluation détaillée de chaque modèle en se concentrant sur :

* les statistiques basées sur les déciles,
* la courbe ROC et le score AUC,
* le coefficient de Gini,
* la statistique KS.

**Processus :**

* Calculer les probabilités prédites,
* regrouper ces probabilités en déciles et calculer les taux d’événements,
* tracer la courbe ROC et calculer les métriques d’évaluation.

---

### **6. Interprétabilité avec SHAP et LIME**

```python
import shap

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_encoded)

shap.summary_plot(shap_values, X_test_encoded)
```

**Explication :**

**Objectif :** Expliquer le modèle XGBoost à l’aide de SHAP.

**Processus :**

* Initialiser l’explainer SHAP,
* générer les valeurs SHAP pour les données de test,
* afficher l’importance globale des features avec `summary_plot`.

---

```python
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_train_balanced_under.values,
    feature_names=X_train_balanced_under.columns,
    class_names=['Non-Default', 'Default'],
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(
    data_row=X_test_encoded.iloc[0].values,
    predict_fn=xgb_model.predict_proba
)

lime_exp.show_in_notebook(show_table=True)
```

**Explication :**

**Objectif :** Utiliser LIME pour générer des explications spécifiques à une instance.

**Processus :**

* Initialiser LIME avec les données d’entraînement et les noms des features,
* produire une explication pour une instance du test,
* visualiser l’explication dans le notebook.

---
