# Modélisation Du Risque De Crédit

## Vue d’ensemble du projet

La Modélisation Du Risque De Crédit est essentielle pour les institutions financières afin d’évaluer la probabilité qu’un emprunteur fasse défaut sur un prêt. Ce projet implique l’analyse de plusieurs jeux de données pour identifier les facteurs influençant le risque de crédit, conduisant à de meilleures décisions. Le projet fournit un système d’évaluation du risque de crédit basé sur le machine learning. Il évalue le risque de défaut des emprunteurs, calcule des scores de crédit et attribue des notes de crédit. Le projet est développé en Python et Streamlit, offrant une interface interactive et conviviale.

[**Lien Web**](https://credit-risk-modeling-lauki-finance.streamlit.app/)

## Prédiction du risque de défaut : Évaluation et déploiement du modèle

### Vue d’ensemble

Ce projet vise à développer un modèle de machine learning pour prédire le risque de défaut, en garantissant haute précision et interprétabilité. Le modèle final utilise des techniques avancées pour fournir des informations exploitables, le rendant adapté au déploiement en conditions réelles.

### Fonctionnalités clés

* **Jeu de données** : problème de classification déséquilibré avec 10 % de défauts.
* **Techniques utilisées** :

  * Feature engineering basé sur la pertinence métier et l’analyse statistique.
  * Méthodes de rééchantillonnage (sur-échantillonnage via SMOTE, sous-échantillonnage).
* **Modèles évalués** :

  * Régression logistique
  * Random Forest
  * XGBoost

### Modèle sélectionné

* **Modèle** : XGBoost avec hyperparamètres optimisés via Optuna et sous-échantillonnage.
* **Metrics** :

  * AUC : 0,98
  * Coefficient de Gini : 0,97
  * Statistique KS : 86,87 %
* **Outils d’interprétabilité** :

  * SHAP (importance des features)

    ![FI](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Feature%20importance.png)

  * LIME (interprétabilité locale)

    ![lime](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Lime.JPG)

### Résultats clés

* Le modèle montre une excellente capacité à classer les défauts avec précision et rappel élevés.
* L’analyse par déciles confirme une séparation claire des instances à haut risque.

#### Prêt pour le déploiement

* **Points forts** :

  * Performance élevée sur toutes les métriques
  * Interprétabilité assurant la conformité aux exigences métier et réglementaires
* **Stratégies d’atténuation** : gérer les risques liés au sous-échantillonnage via des réentraînements périodiques.

### Visualisations

1. Courbe AUC-ROC avec performance quasi parfaite (AUC : 0,99)

   ![rocauc](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/ROC%20Curve.png)

2. Graphique récapitulatif SHAP montrant les features ayant le plus d’influence sur les prédictions.

### Mode d’utilisation

1. **Entraîner le modèle** : Scripts pour prétraitement des données, entraînement et optimisation des hyperparamètres inclus.
2. **Évaluer le modèle** : Outils pour générer les métriques, analyse par déciles et graphiques d’interprétabilité.
3. **Déployer le modèle** : Pipeline de déploiement préconstruit pour intégration dans les systèmes métiers.

### Pourquoi ce projet se distingue

* Combine des techniques de machine learning de pointe avec l’interprétabilité.
* Résout un problème métier réel avec rigueur et précision.
* Offre un chemin clair du développement du modèle jusqu’au déploiement.

---

# **Exécution du projet : Modélisation Du Risque De Crédit**

## **Fonctionnalités**

* **Évaluation interactive du risque de crédit** : saisie des informations sur l’emprunteur et le prêt avec prédictions en temps réel.
* **Machine Learning avancé** : utilisation d’un modèle XGBoost finement ajusté pour des prédictions robustes et précises.
* **Design évolutif** : structure modulaire avec utilitaires réutilisables et optimisation des hyperparamètres.

---

## **Structure du répertoire du projet**

```
project-root/
│
├── model/
│   ├── model_data.pkl                # Modèle ML sérialisé et données de prétraitement
│   ├── tuned_hyperparameters.txt    # Détails des hyperparamètres optimisés
│
├── Jeff Finance.jpg                  # Logo du projet ou image associée
├── Readme.md                         # Fichier de documentation
├── main.py                           # Fichier de l’application Streamlit
├── requirements.txt                  # Liste des packages Python requis
├── utils.py                          # Fonctions utilitaires pour prédiction et prétraitement
```

---

## **Guide d’installation**

### **Étape 1 : Cloner le dépôt**

Téléchargez le dépôt du projet sur votre machine :

```bash
git clone https://github.com/username/repository-name.git
cd repository-name//project-root
```

### **Étape 2 : Configurer l’environnement Python**

Assurez-vous d’avoir Python 3.8 ou supérieur. Il est recommandé d’utiliser un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate    # Sur macOS/Linux
venv\Scripts\activate       # Sur Windows
```

### **Étape 3 : Installer les dépendances**

Installez tous les packages Python listés dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

### **Étape 4 : Lancer l’application**

Démarrez l’application Streamlit avec la commande suivante :

```bash
streamlit run main.py
```

---

## **Mode d’utilisation**

1. Ouvrez l’URL affichée dans le terminal après `streamlit run main.py` (typiquement `http://localhost:8501/`).
2. Utilisez l’interface interactive pour :

   * Entrer les informations de l’emprunteur (âge, revenu, montant du prêt, etc.)
   * Ajuster les curseurs et menus déroulants pour d’autres paramètres
   * Cliquer sur “Calculate Risk” pour afficher :

     * Probabilité de défaut
     * Score de crédit
     * Note de crédit
3. Consultez les analyses et recommandations fournies.

---

## **Notes supplémentaires**

### **Dépendances**

Le projet requiert les bibliothèques suivantes :

* `streamlit` : pour l’interface web interactive
* `scikit-learn` : pour le prétraitement et la gestion du modèle
* `joblib` : pour charger le modèle sérialisé
* `pandas` et `numpy` : pour la manipulation des données
* `xgboost` et autres

Toutes les dépendances sont listées dans `requirements.txt`.

### **Personnalisation**

* Pour utiliser un autre modèle ML, remplacez `model_data.pkl` par votre modèle sérialisé et ajustez les features dans `utils.py`.
* Modifiez l’interface dans `main.py` si les entrées ou sorties changent.

---

## **Exemples de captures d’écran**

1. **Page d’accueil** : affiche le titre du projet et l’interface de saisie.
2. **Page des résultats** : montre la probabilité de défaut, le score de crédit et la note avec des recommandations exploitables.

![image](https://github.com/user-attachments/assets/d1b51282-cf2a-4e9a-ab19-fbe407b425ba)

---
