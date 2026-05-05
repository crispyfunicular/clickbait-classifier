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

- Fichier utilitaire : `feature_utils.py` (fonctions partagées, notamment les features manuelles ; nécessaire pour pouvoir recharger les modèles sauvegardés avec `joblib`)

#### Étape 1 - Exploration des données (EDA)

- Script : `1_eda.py` (`python scripts/1_eda.py` depuis la racine du dépôt)
- Charger le CSV avec `pandas` ou `polars`
- Vérifier la distribution des classes et les longueurs de titres
- Analyser les mots fréquents et n-grammes caractéristiques par classe
- Visualiser quelques exemples représentatifs de chaque classe

#### Étape 2 - Prétraitement et vectorisation

- Script : `2_features.py` (`python scripts/2_features.py` depuis la racine du dépôt)
- Séparation train/test stratifiée : `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`
- **Vectorisation 1** — Bag of Words (`CountVectorizer`)
- **Vectorisation 2** — TF-IDF (`TfidfVectorizer`)
- (Optionnel) Traits linguistiques manuels : longueur du titre, présence de chiffres, points de suspension, pronoms interrogatifs, majuscules…

> Le vectoriseur doit être entraîné (`fit`) uniquement sur le train, puis appliqué (`transform`) sur le test.

#### Étape 3 - Mise en place des algorithmes de classification

Les consignes imposent 2 des 3 algorithmes suivants :

- Script : `3_models.py` (`python scripts/3_models.py` depuis la racine du dépôt)
- **Naive Bayes** — `MultinomialNB`
- **SVM** — `LinearSVC` ou `SVC` (kernel RBF)
- **Arbre de décision** — `DecisionTreeClassifier` (équivalent de J48)

Optionnels pour enrichir la comparaison : `LogisticRegression`, `RandomForestClassifier`

Utiliser des `Pipeline` scikit-learn pour combiner vectorisation et classifieur, afin d'éviter toute fuite de données.

#### Étape 4 - Évaluation

- Script : `4_evaluate.py` (`python scripts/4_evaluate.py` depuis la racine du dépôt)
- Métriques : accuracy, precision, recall, F1 (`classification_report`)
- Validation croisée (k=5) avec `cross_val_score`
- Recherche d'hyperparamètres avec `GridSearchCV`
- Tableau de synthèse : accuracy + F1 macro par combinaison modèle × vectorisation
- Sorties : `artifacts/step4/` (rapport texte, scores CSV, matrice de confusion en PNG)

#### Étape 5 - Analyse et commentaires

- Script : `5_analysis.py` (`python scripts/5_analysis.py` depuis la racine du dépôt)
- Matrice de confusion par modèle
- Exemples de titres mal classés
- Traits les plus discriminants (`coef_` des modèles linéaires)
- Discussion : quel modèle performe le mieux et pourquoi ?
- Sorties : `artifacts/step5/` (faux positifs/négatifs en CSV, traits discriminants en TXT)

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


## Résultats

Les résultats ci-dessous proviennent des scripts `scripts/3_models.py`, `scripts/4_evaluate.py` et `scripts/5_analysis.py`.

- **Meilleur pipeline (test)** : `LinearSVC | TF-IDF + manuels`
  - **Accuracy (test)** : 0.9842
  - **F1 macro (test)** : 0.9842
  - **Détail (test, rapport de classification)** :
    - Classe 0 (non-clickbait) : precision 0.986 / recall 0.982 / F1 0.984 (support 3200)
    - Classe 1 (clickbait) : precision 0.982 / recall 0.987 / F1 0.984 (support 3200)

- **Comparaison rapide des meilleurs modèles (F1 macro sur test)** :
  - `LinearSVC | TF-IDF + manuels` : 0.9842
  - `LinearSVC | BoW + manuels` : 0.9783
  - `MultinomialNB | BoW + manuels` : 0.9644

- **Traits discriminants (extraits, modèle linéaire)** :
  - Lecture des noms de traits :
    - `text::...` = un mot / n-gramme appris par le vectoriseur (ex: `text::obama` correspond au token “obama”)
    - `manual::...` = un trait « fait main » calculé à partir du texte brut (ex: `manual::upper_ratio` = proportion de lettres majuscules dans le titre)
  - Plutôt associés à **clickbait (classe 1)** : `manual::upper_ratio`, `text::guess`, `text::tell`, `text::things`, `text::identify`, `text::celebrities`…
  - Plutôt associés à **non-clickbait (classe 0)** : `text::wins`, `text::obama`, `text::leader`, `text::china`, `text::economy`, `text::says`, `text::deal`…

Fichiers produits :
- `artifacts/step3/summary.csv` (tableau complet des scores modèle × vectorisation)
- `artifacts/step4/` (scores + rapport + matrice de confusion)
- `artifacts/step5/` (faux positifs/négatifs + traits discriminants)
