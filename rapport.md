# Traitement statistique des données - Clickbait / pièges à clics

> Lena Baraquin & Morgane Bona-Pellissier (Maser 1 pluriTAL)

---

## Objectifs

- Classifier automatiquement des titres d’articles en anglais en deux catégories : **non-clickbait** (0) vs **clickbait** (1) ;
- comparer plusieurs algorithmes (consigne : au moins deux parmi Naive Bayes, SVM, arbre de décision) ;
- commenter les performances et les erreurs typiques.

---

## Données et ressources

| Élément | Détail |
|--------|--------|
| Fichier | `clickbait_data.csv` (UTF-8) |
| Source | jeu Kaggle (https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) |
| Contenu | une ligne = un titre ; colonnes `headline` (texte), `clickbait` (0 ou 1) |
| Volume total | 32 000 titres |
| Répartition | 16 001 non-clickbait / 15 999 clickbait -> corpus équilibré |
| Taille des documents | titres courts uniquement (pas d’article complet) ; longueurs analysées en EDA (`scripts/1_eda.py`) |

---

## Méthodologie (aperçu)

- **Découpage** : *train* / *test* 80 % / 20 %, stratifié sur le label, `random_state=42` (reproductibilité) — `scripts/2_features.py` (même logique dans `3_models.py`).
```python
# scripts/2_features.py — aperçu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **Fuites** : vectorisation et modèle dans des pipelines scikit-learn ; le vectoriseur est ajusté uniquement sur le *train* — `scripts/3_models.py`.
```python
# scripts/3_models.py — aperçu
if not add_manual:
    return Pipeline([("vect", vectorizer), ("clf", classifier)])
features = FeatureUnion([("text", vectorizer),
                         ("manual", FunctionTransformer(manual_features, validate=False))])
return Pipeline([("features", features), ("clf", classifier)])
```
- **Traits textuels** : sac de mots et bigrammes (`CountVectorizer`), TF-IDF (`TfidfVectorizer`), plus **traits manuels** optionnels (longueur, chiffres, suspension de points, etc. — `feature_utils.py`).
```python
# scripts/feature_utils.py — extraits
has_number = s.str.contains(r"\d", regex=True).to_numpy(dtype=float)
has_ellipsis = s.str.contains(r"\.\.\.|…", regex=True).to_numpy(dtype=float)
n_question = s.str.count(r"\?").to_numpy(dtype=float)

# scripts/3_models.py — exemple de vectoriseur (TF-IDF, mêmes idées pour BoW)
TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
```
- **Validation croisée** : **5 folds** stratifiés sur le *train* pour comparer les modèles (`scripts/3_models.py`).
```python
# scripts/3_models.py — evaluate_model (extrait)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(pipe, split.X_train, split.y_train, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
pipe.fit(split.X_train, split.y_train)
y_pred = pipe.predict(split.X_test)
```
- **Évaluation finale** : métriques sur le jeu de *test* uniquement (`scripts/4_evaluate.py`).
```python
# scripts/4_evaluate.py — aperçu
model = load(MODEL_PATH)
y_pred = model.predict(split.X_test)
accuracy_score(split.y_test, y_pred)
f1_score(split.y_test, y_pred, average="macro")
classification_report(split.y_test, y_pred, digits=3)
```

---

## Expériences réalisées

Modèles testés (parmi autres) : **MultinomialNB**, **LinearSVC**, **DecisionTreeClassifier**, combinés à BoW ou TF-IDF, avec ou sans traits manuels. Hyperparamètres de référence : ex. `LinearSVC(C=1.0, max_iter=20000)`, `MultinomialNB(alpha=1.0)`, arbre avec profondeur et `min_samples_split` fixés pour limiter le sur-apprentissage. Tableau complet : `artifacts/step3/summary.csv`.

---

## Résultats quantitatifs

**Meilleur pipeline** (sélectionné par F1 macro sur le test à l’étape 3) : **LinearSVC | TF-IDF + traits manuels**.

| Indicateur | Valeur (jeu de test, *n* = 6 400) |
|------------|-------------------------------------|
| Accuracy | **0,9842** |
| F1 macro | **0,9842** |

**Par classe** (rapport de classification sur le test) :

| Classe | Précision | Rappel | F1 |
|--------|-----------|--------|-----|
| 0 (non-clickbait) | 0,986 | **0,982** | 0,984 |
| 1 (clickbait) | 0,982 | **0,987** | 0,984 |

**Autres modèles** (aperçu — détail dans `summary.csv`) : MultinomialNB et arbres sont nettement au-dessous du SVM linéaire sur ce corpus ; les arbres sans traits manuels ou avec TF-IDF seuls peuvent fortement se dégrader.

---

## Commentaire et interprétation (brouillon)

- Les deux classes obtiennent des scores très proches ; le **rappel** est légèrement plus faible pour la classe **0** que pour la **1** : un peu plus d’erreurs où un titre réel est pris pour du clickbait que l’inverse.  
- Les **faux positifs** (ex. titres factuels avec formulation interrogative ou « accroche ») montrent que la frontière n’est pas purement lexicale : la presse utilise aussi des formulations « engageantes ».  
- Les **faux négatifs** incluent souvent des titres **très courts** ou ambigus (*Memory Loss*, *The Tennis Racket*), où les indices associés au clickbait dans le corpus sont peu présents.  
- Les **coefficients** du `LinearSVC` (`artifacts/step5/top_features.txt`) associent plutôt au clickbait des marqueurs de style (majuscules, termes type quiz/listicle) et au non-clickbait un vocabulaire plus « une factuelle » (économie, politique, etc.). Il s'agit des corrélations observées sur ce jeu et non d'une définition universelle du journalisme. La généralisation à d'autres sources n'est donc pas garantie.
- **Prudence** : les scores ~98 % sont élevés (cohérents avec un corpus binaire bien séparé)

### Matrice de confusion (jeu de test)

![Matrice de confusion — lignes : vrai label, colonnes : prédiction](artifacts/step4/confusion_matrix.png)

### Exemples d’erreurs (fichiers `artifacts/step5/false_positives.csv` et `false_negatives.csv`)

**Faux positifs** : vrai label *non-clickbait* (0), prédiction *clickbait* (1)

| Titre |
|-------|
| What Is Facebook Actually Worth? |
| Star Wars III premieres at Cannes |
| Happy Gilmore Was On to Something |

**Faux négatifs** : vrai label *clickbait* (1), prédiction *non-clickbait* (0)

| Titre |
|-------|
| Memory Loss |
| The Tennis Racket |
| Whose Concert Tour Should You Open For |
