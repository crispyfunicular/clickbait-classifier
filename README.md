# clickbait-classifier
Projet de master de Machine Learning : Classification de titres d'articles (Clickbait vs Non-Clickbait) via des techniques de NLP avec Scikit-Learn.

## Consignes :
1. Décider d'une thématique générale
2. Trouver un jeu de données et un certain nombre de catégories
3. Mettre en place des algorithmes de classification et d'évaluation avec Weka ou scikit
4. Faire une évaluation
5. Commenter le résultat

## Feuille de route

> Les étapes 1 et 2 des consignes sont déjà réalisées : thématique choisie (clickbait vs. authentique), jeu de données trouvé sur Kaggle.

### Corpus

- Fichier `clickbait_data.csv` : 32 000 titres en anglais
- Colonnes : `headline` (texte brut) et `clickbait` (0 = authentique, 1 = clickbait)
- Classes équilibrées : 16 001 non-clickbait / 15 999 clickbait — pas besoin de rééchantillonnage

### Scripts
Tous les scripts `.py` se trouvent dans le dossier `scripts/`

#### Étape 1 - Exploration des données (EDA) -> fait

- Script : `1_eda.py` (`python scripts/1_eda.py` depuis la racine du dépôt)
- Charger le CSV avec `pandas` ou `polars`
- Vérifier la distribution des classes et les longueurs de titres
- Analyser les mots fréquents et n-grammes caractéristiques par classe
- Visualiser quelques exemples représentatifs de chaque classe

#### Étape 2 - Prétraitement et vectorisation -> fait

- Script : `2_features.py` (`python scripts/2_features.py` depuis la racine du dépôt)
- Séparation train/test stratifiée : `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`
- **Vectorisation 1** — Bag of Words (`CountVectorizer`)
- **Vectorisation 2** — TF-IDF (`TfidfVectorizer`)
- (Optionnel) Traits linguistiques manuels : longueur du titre, présence de chiffres, points de suspension, pronoms interrogatifs, majuscules…

> Le vectoriseur doit être entraîné (`fit`) uniquement sur le train, puis appliqué (`transform`) sur le test.

#### Étape 3 - Mise en place des algorithmes de classification -> fait

Les consignes imposent 2 des 3 algorithmes suivants :

- Script : `3_models.py` (`python scripts/3_models.py` depuis la racine du dépôt)
- **Naive Bayes** — `MultinomialNB`
- **SVM** — `LinearSVC` ou `SVC` (kernel RBF)
- **Arbre de décision** — `DecisionTreeClassifier` (équivalent de J48)

Optionnels pour enrichir la comparaison : `LogisticRegression`, `RandomForestClassifier`

Utiliser des `Pipeline` scikit-learn pour combiner vectorisation et classifieur, afin d'éviter toute fuite de données.

#### Étape 4 - Évaluation

- Métriques : accuracy, precision, recall, F1 (`classification_report`)
- Validation croisée (k=5) avec `cross_val_score`
- Recherche d'hyperparamètres avec `GridSearchCV`
- Tableau de synthèse : accuracy + F1 macro par combinaison modèle × vectorisation

#### Étape 5 - Analyse et commentaires

- Matrice de confusion par modèle
- Exemples de titres mal classés
- Traits les plus discriminants (`coef_` des modèles linéaires)
- Discussion : quel modèle performe le mieux et pourquoi ?

### Compte-rendu

Sections obligatoires :

- Objectifs du projet
- Description des données (origine, format, statut juridique, distribution)
- Méthodologie (étapes, choix techniques, reproductibilité)
- Expériences réalisées (paramètres, mode de calcul)
- Résultats et discussion

### Livrables

- Scripts ou notebooks (`1_eda.py`, puis `2_features` / `3_models` selon votre choix)
- `clickbait_data.csv`
- `Nom1_Prenom1-Nom2_Prenom2.pdf`
- Archive `.zip` avec tout le contenu, encodé en UTF-8

### Points de vigilance

- Ne jamais évaluer sur les données d'entraînement (`random_state=42` pour la reproductibilité)
- Se méfier des résultats "trop beaux" : les justifier si F1 > 0.95
- Tous les fichiers textes en UTF-8
