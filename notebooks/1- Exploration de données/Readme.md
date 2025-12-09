# Exploration et Familiarisation avec les Données

## Objectifs

1. Charger et explorer trois jeux de données : données bureau(bureau data), données de prêt(loan data) et données clients(customer data).
2. Se familiariser avec la structure, le contenu et les caractéristiques clés de chaque dataset.
3. Fusionner les jeux de données dans une structure unifiée en utilisant le `Customer ID` comme clé primaire.
4. Préparer le dataset fusionné pour une modélisation prédictive et une analyse plus approfondie.

## Description des Jeux de Données

### 1. Données Bureau (bureau data)

Contiennent des détails sur l’historique financier des clients auprès d’autres institutions financières. Colonnes clés :

* **Customer ID** : Identifiant unique de chaque client.
* **Credit Amount** : Montant total du crédit emprunté.
* **Payment History** : Historique des paiements passés.

### 2. Données de Prêt (loan data)

Incluent des informations sur les prêts accordés par l’institution financière. Colonnes clés :

* **Customer ID** : Identifiant unique de chaque client.
* **Loan Amount** : Montant du prêt.
* **Loan Status** : Indique si le prêt est actif, clôturé ou en défaut.

### 3. Données Clients (customer data)

Fournissent des informations démographiques et personnelles des clients. Colonnes clés :

* **Customer ID** : Identifiant unique de chaque client.
* **Age** : Âge du client.
* **Income** : Revenu annuel du client.

## Processus de Traitement

### 1. Chargement des Données

Les jeux de données ont été chargés à l’aide de la bibliothèque `pandas` pour une gestion flexible de grandes quantités de données.

#### Extrait de Code :

```python
import pandas as pd

# Load datasets
bureau_data = pd.read_csv('bureau.csv')
loan_data = pd.read_csv('loan.csv')
customer_data = pd.read_csv('customer.csv')
```

**Explication** : La fonction `read_csv` charge les fichiers CSV dans des DataFrames Pandas pour l’analyse ultérieure.

### 2. Exploration des Données

Une Analyse Exploratoire des Données (EDA) a été réalisée pour comprendre les datasets. Cela inclut :

* Afficher les premières lignes avec `.head()`.
* Vérifier les valeurs manquantes via `.isnull().sum()`.
* Analyser la distribution des colonnes clés avec `.describe()`.

#### Extrait de Code :

```python
# Display summary statistics
print(bureau_data.describe())
print(loan_data.describe())
print(customer_data.describe())

# Check for missing values
print(bureau_data.isnull().sum())
```

**Explication** : Les statistiques descriptives et le décompte des valeurs manquantes donnent une idée de la qualité et de la distribution des données.

### 3. Fusion des Données

Les datasets ont été fusionnés en un DataFrame unique en utilisant la fonction `merge` sur la colonne `Customer ID`.

#### Extrait de Code :

```python
# Merge datasets
data_merged = bureau_data.merge(loan_data, on='Customer_ID', how='inner')
data_merged = data_merged.merge(customer_data, on='Customer_ID', how='inner')
```

**Explication** : La fusion permet de combiner toutes les informations pertinentes liées aux clients, aux prêts et à l’historique bureau.

### 4. Nettoyage des Données

Après la fusion, le nettoyage inclut :

* La gestion des valeurs manquantes.
* La suppression de colonnes non pertinentes.
* Le renommage de colonnes pour plus de clarté.

**Explication** : Le nettoyage garantit que le dataset est prêt pour une analyse avancée et la modélisation.

## Insights

* **Modèles de Crédit** : Les clients ayant des revenus plus élevés présentent de meilleurs historiques de crédit.
* **Statut des Prêts** : Les taux de défaut sont plus élevés pour les prêts dépassant un certain montant.
* **Démographie** : Les clients plus jeunes (<30 ans) enregistrent des taux de défaut plus élevés.

## Étapes Futures

1. Effectuer du feature engineering pour créer des métriques dérivées comme le ratio dette/revenu.
2. Construire des modèles prédictifs (ex. régression logistique, forêts aléatoires) pour classifier le risque crédit.
3. Visualiser les insights clés avec des bibliothèques avancées comme `matplotlib` et `seaborn`.

## Outils et Technologies

* **Python** : Manipulation et analyse des données.
* **Pandas** : Chargement, fusion et nettoyage.
* **Matplotlib & Seaborn** : Visualisation des données.
* **Scikit-learn** (à venir) : Construction et évaluation des modèles.

## Conclusion

Cette partie pose les bases d’une analyse complète du risque crédit. Elle met l’accent sur la compréhension et la préparation des données avant l’application de techniques prédictives, garantissant ainsi précision et interprétabilité des résultats.




