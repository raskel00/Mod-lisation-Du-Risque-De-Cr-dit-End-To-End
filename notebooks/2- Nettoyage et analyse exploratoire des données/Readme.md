# Nettoyage des Données de Risque Crédit et Analyse Exploratoire des Données (EDA)

Ce dépôt illustre un processus complet de **nettoyage des données** et d’**analyse exploratoire des données (EDA)** pour un dataset de risque crédit. Le projet se concentre sur la gestion des incohérences de données, des valeurs aberrantes et des valeurs manquantes, tout en extrayant des insights significatifs via l’analyse univariée et bivariée. Ce travail vise à servir de base pour des applications avancées en machine learning, telles que la prédiction des défauts de paiement de prêts.

---

## Présentation du Projet

L’objectif de ce projet est de préparer un dataset de risque crédit pour l’analyse et de développer une compréhension approfondie de ses caractéristiques. Le projet est divisé en sections clés :

1. **Nettoyage des Données** :

   * Gestion des valeurs manquantes.
   * Identification et correction des données incohérentes ou inappropriées.
   * Vérification des types de données et détection des doublons.
   * Gestion des valeurs aberrantes en utilisant des techniques statistiques et des connaissances métier.

2. **Ingénierie des Features** :

   * Création de nouvelles features pour obtenir de meilleurs insights.
   * Validation et filtrage des données selon des règles spécifiques au domaine.

3. **Analyse Exploratoire des Données (EDA)** :

   * Analyse univariée pour étudier la distribution des variables individuelles.
   * Analyse bivariée pour explorer les relations entre les variables et la variable cible (`default`).
   * Analyse de corrélation pour comprendre la force des relations entre variables numériques.

---

## Déroulé du Code

Ci-dessous une explication détaillée du code et de son objectif :

---

### **1. Chargement des Données**

```python
df = pd.read_csv("explored_data.csv")
df.head()
```

* Le dataset est chargé dans un DataFrame pandas pour le traitement.
* Les premières lignes sont inspectées afin de comprendre la structure et le contenu.

---

### **2. Gestion des Valeurs Manquantes**

```python
df.isnull().sum()
df.dropna(inplace=True)
```

* Les valeurs manquantes sont vérifiées avec `isnull().sum()`.
* Les valeurs manquantes sont supprimées car leur proportion était négligeable.

---

### **3. Correction des Incohérences dans les Variables Catégorielles**

```python
df['loan_purpose'] = df['loan_purpose'].replace({'Personaal': 'Personal'})
```

* Les incohérences dans les variables catégorielles (ex. fautes de frappe) sont corrigées avec `.replace()`.

---

### **4. Correction des Types de Données**

```python
df['zipcode'] = df['zipcode'].astype(str)
df['disbursal_date'] = pd.to_datetime(df['disbursal_date'])
df['installment_start_dt'] = pd.to_datetime(df['installment_start_dt'])
```

* Les colonnes comme `zipcode` sont converties en chaînes, et les colonnes de date sont converties au format datetime.

---

### **5. Vérification des Doublons**

```python
df.duplicated().sum()
```

* Les doublons sont identifiés ; aucun doublon n’a été trouvé dans ce dataset.

---

### **6. Gestion des Valeurs Aberrantes**

#### **Méthode IQR**

```python
def iqr(column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    IQR = q3 - q1
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)
    print(f"Borne inférieure : {lower_bound} et Borne supérieure : {upper_bound}\n")
```

* La plage interquartile (IQR) est utilisée pour détecter les valeurs aberrantes.
* Chaque colonne numérique est vérifiée itérativement.

#### **Limitation des Valeurs Extrêmes**

```python
df = df[df['income'] <= df['income'].quantile(0.99)]
```

* Les valeurs extrêmes sont limitées avec le seuil du 99e percentile afin de conserver les données pertinentes.

#### **Validations Spécifiques au Domaine**

```python
df['valid_gst'] = df['gst'] <= (df['loan_amount'] * 0.20)
df['valid_net_disbursement'] = df['net_disbursement'] <= (df['loan_amount'] - df['gst'])
df['valid_principal_outstanding'] = df['principal_outstanding'] <= df['loan_amount']
df['valid_bank_balance'] = df['bank_balance_at_application'] >= 0
```

* Des validations supplémentaires sont appliquées selon les règles du domaine (ex. GST limité à 20% du montant du prêt).

---

### **7. Analyse Univariée**

#### **Variables Catégorielles**

```python
cat_cols = ['gender', 'marital_status', 'employment_status', 'residence_type', 'loan_purpose', 'loan_type', 'default']
sns.countplot(data=df, x=column, ax=axes[i], palette='Set2')
```

* Des diagrammes en barres sont créés pour visualiser la distribution des variables catégorielles.

#### **Variables Numériques**

```python
sns.boxplot(data=df, y=column, ax=axes[i], palette='Set2')
```

* Des boxplots sont générés pour visualiser la dispersion et détecter les valeurs aberrantes potentielles.

---

### **8. Analyse Bivariée**

#### **Scatter Plots**

```python
sns.scatterplot(data=df, x='age', y='income', hue='default', palette='Set2')
```

* Les scatterplots montrent les relations entre variables numériques, comme `age` et `income`, colorées par défaut.

#### **Count Plots**

```python
sns.countplot(data=df, x='loan_purpose', hue='default', palette='Set2')
```

* Les count plots examinent la distribution des variables catégorielles par rapport à la variable cible `default`.

#### **Heatmaps**

```python
crosstab = np.round(pd.crosstab(df['employment_status'], df['default'], normalize='index'), 2)
sns.heatmap(crosstab, annot=True, cmap='coolwarm')
```

* Les tableaux croisés sont visualisés via des heatmaps pour identifier des patterns dans les variables catégorielles.

---

### **9. Analyse de Corrélation**

#### **Matrice de Corrélation**

```python
plt.figure(figsize=(20, 20))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
```

* Les corrélations entre variables numériques sont analysées pour détecter la multicolinéarité et identifier des relations prédictives.

#### **Corrélation avec la Variable Cible**

```python
target_correlation = correlation_matrix['default'].sort_values(ascending=False)
sns.barplot(x=target_correlation.index, y=target_correlation.values, palette='coolwarm')
```

* Les variables les plus corrélées avec `default` sont mises en évidence pour guider la sélection des features.

---

### **10. Résumé**

Un résumé détaillé est fourni à la fin de l’analyse, mettant en avant les principaux résultats et garantissant l’intégrité des données pour la modélisation future.

#### **Processus de Gestion des Valeurs Aberrantes**

Ce processus combine des méthodes techniques et la connaissance métier pour assurer l’intégrité des données :

1. **Méthode IQR** : Identification et traitement des outliers dans les colonnes numériques.
2. **Filtrage par Quantiles** : Limitation des valeurs extrêmes via les percentiles.
3. **Features Dérivées** : Calcul de `diff` (différence entre montant du prêt et frais de traitement) et `pct` (pourcentage des frais de traitement) pour des insights supplémentaires.
4. **Validations Métier** :

   * GST limité à 20% du montant du prêt.
   * Net disbursement ≤ montant du prêt - GST.
   * Principal outstanding ≤ montant du prêt.
   * Solde bancaire à la demande ≥ 0.
5. **Vérification Composite** : Combinaison de plusieurs règles en un indicateur de validité (`valid_loan`) pour assurer la cohérence globale.
6. **Résultat Final** :

   * Nombre de prêts invalides : **0**

Cette approche garantit une gestion efficace des anomalies statistiques et des incohérences spécifiques au domaine.

---

#### **Analyse Univariée**

**1. Distribution des Variables Catégorielles**
Les diagrammes montrent la distribution des variables catégorielles du dataset. La majorité des emprunteurs sont des hommes mariés, principalement salariés. Les maisons possédées sont le type de résidence le plus courant, et les prêts auto sont les plus fréquents. Les prêts garantis sont plus populaires que les non garantis. La variable cible `default` montre que la majorité des prêts n’ont pas été en défaut.

**2. Distribution des Variables Numériques**
Les boxplots montrent que la plupart des variables présentent une distribution asymétrique à droite, concentrant la majorité des valeurs vers le bas. Certaines variables comme "Loan amount" et "Processing fee" sont plus uniformes.

---

#### **Analyse Bivariée**

**1. Montant du Prêt vs Revenu par Défaut**
Scatterplot indiquant que les prêts plus élevés correspondent souvent à des revenus plus élevés. Malgré cela, la majorité des prêts ont été remboursés, mais les défauts surviennent davantage chez les revenus élevés.

**2. Montant du Prêt vs Âge par Défaut**
Les emprunteurs sont majoritairement âgés de 20 à 60 ans. Les plus âgés ont tendance à contracter des prêts plus élevés, mais la majorité des prêts, quel que soit l’âge, ne sont pas en défaut.

**3. Âge vs Revenu par Défaut**
Scatterplot montrant la relation entre âge et revenu. Les défauts apparaissent sur une large gamme d’âges et de revenus.

**4. Objet du Prêt vs Défaut**
Bar chart montrant que la majorité des prêts ne sont pas en défaut. Les prêts immobiliers ont le plus de défauts, les prêts auto le moins.

**5. Type de Prêt vs Défaut**
Les prêts garantis sont plus fréquents et présentent moins de défauts que les prêts non garantis.

**6. Proportion de Défauts par Statut Professionnel**
Heatmap montrant que salariés et indépendants ont une majorité de prêts non défaillants, avec un taux de défaut légèrement plus élevé pour les indépendants.

**7. Proportion de Défauts par État Civil**
Heatmap montrant que les personnes mariées ont un taux de défaut légèrement inférieur à celui des célibataires.

**8. Défauts par Ville**
Bar chart indiquant que toutes les villes ont plus de prêts non défaillants que de défauts, Mumbai ayant le plus grand nombre de prêts dans les deux catégories.

**9. Analyse de Corrélation**
La matrice de corrélation montre des relations positives fortes entre montant du prêt, montant accordé, frais de traitement, GST, net disbursement et principal outstanding. Revenu et montant du prêt sont aussi positivement corrélés. L’âge et le revenu présentent une corrélation modérée positive. Le ratio d’utilisation du crédit est modérément corrélé avec les mois en retard et le DPD total.

**10. Corrélation des Variables Numériques avec Default**
Le ratio d’utilisation du crédit est fortement lié au risque de défaut, tandis que la durée des prêts a l’effet inverse. D’autres facteurs comme le montant du prêt ou la durée de l’historique de crédit ont peu d’impact apparent.

---

## Points Clés à Retenir

Ce projet :

* Montre la capacité à nettoyer et prétraiter efficacement les données.
* Utilise des techniques statistiques et spécifiques au domaine pour gérer et valider les outliers.
* Extrait des insights précieux via la visualisation et l’analyse de corrélation.
* Établit une base solide pour la modélisation prédictive dans l’analyse du risque crédit.

