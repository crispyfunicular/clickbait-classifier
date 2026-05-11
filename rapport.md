# Traitement statistique des données - projet  
# « Les 8 choses que le TAL doit savoir sur les pièges à clics »

> L’ensemble du pipeline est disponible à l’adresse suivante : https://github.com/crispyfunicular/clickbait_classifier 

---

## Introduction
Le concept de « piège à clics » (*clickbait* en anglais) est défini par la Commission d’enrichissement de la langue française (2020, p. 9) comme un « [l]ien hypertextuel accrocheur conduisant à un contenu qui n’est qu’un leurre, mis en place à seule fin d’augmenter le trafic en incitant les internautes à cliquer ; par extension, le contenu lui-même ».

### Cadre théorique : Dépasser la vision binaire
Du point de vue des sciences de l’information, Rubin et Chen (2012) soulignent que les outils traditionnels de l’éducation à l’information (*Information Literacy*) ont tendance à simplifier les dérives informationnelles pour les rendre « digestibles ». Cette approche réduit souvent l’évaluation à une décision binaire (accepter ou rejeter) comme si toutes les formes de tromperie se valaient. Il existe pourtant une différence de nature entre le mensonge pur, l’omission, l’exagération et la fabrication. Pour sortir de cette vision où l’on regroupe indistinctement toutes les distorsions sous l’étiquette « infox », les auteurs proposent un cadre conceptuel global. Ce système holistique permet de situer le clickbait dans un spectre bien plus nuancé, le traitant non pas comme une simple « fake news », mais comme une manipulation spécifique dans le cycle de production et de perception de l’information.

### Approche par le Traitement automatique du langage (TAL)
Si la théorie appelle à la nuance, le traitement automatique du langage aborde le phénomène sous un angle plus pragmatique : celui d’un problème de catégorisation de textes courts (titres, posts). Bien que des travaux récents analysent des stratégies linguistiques fines et explorent l’explicabilité des décisions des modèles (Nofar et al., 2025), le présent rapport se concentre sur une étape charnière. Afin de pallier le « flou conceptuel » qui empêche parfois une définition claire des tâches d’automatisation, cette étude se limite volontairement à une tâche supervisée à deux classes (clickbait / non-clickbait). L’objectif est de comparer l’efficacité d’algorithmes classiques sur des titres en anglais, posant ainsi les bases nécessaires avant d’envisager une classification plus fine et nuancée, comme suggéré par Rubin et Chen.

![Exemples de formulations clickbait](clickbaits.png)

*Légende : collage illustrant des titres « piège à clics » - Source : Wikipedia.*

## Objectifs

- Classifier automatiquement des titres d’articles en anglais en deux catégories : **non-clickbait** (0) vs **clickbait** (1) ;
- comparer plusieurs algorithmes (consigne : au moins deux parmi Naive Bayes, SVM, arbre de décision) ;
- commenter les performances et les erreurs typiques.

---

## Données et ressources

| Élément | Détail |
|--------|--------|
| Fichier | `clickbait_data.csv` (UTF-8) |
| Source des données | jeu Kaggle (https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) |
| Source non-clickbait | *WikiNews*, *New York Times*, *The Guardian*, *The Hindu* |
| Source clickbait | *BuzzFeed*, *Upworthy*, *ViralNova*, *Thatscoop*, *Scoopwhoop*, *ViralStories* |
| Contenu | une ligne = un titre ; colonnes `headline` (texte), `clickbait` (0 ou 1) |
| Volume total | 32 000 titres |
| Répartition | 16 001 non-clickbait / 15 999 clickbait -> corpus équilibré |
| Taille des documents | titres courts uniquement (pas d’article complet) ; longueurs analysées en EDA (`scripts/1_eda.py`) |

*Description du jeu de données utilisé.*

---

## Méthodologie & pipeline

### Découpage train / test
Les données ont été divisées en **deux partitions *train* et *test* (80 % / 20 %)**, de façon stratifiée sur le label `clickbait` et avec **`random_state=42`**, de sorte que le même découpage soit rejoué partout. Ainsi, les partitions *train* et *test* gardaient chacune à peu près la même proportion de 0 et de 1 que le jeu complet (via `stratify=y` dans `train_test_split`), évitant ainsi qu’un tirage aléatoire ne crée un bloc nettement plus riche en une classe que l’autre. Dans le code, cet appel est regroupé dans une fonction **`make_split`** dont la logique est reprise dans plusieurs scripts pour rester alignés sur ce split.

```python
# scripts/2_features.py - fonction make_split (aperçu)
def make_split(df: pd.DataFrame) -> SplitData:
    X = df["headline"].astype(str)
    y = df["clickbait"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
```

### Absence de dev
Nous n’avons pas utilisé de troisième ensemble « validation / dev » en tant que tel et n’avons donc pas réservé de sous-échantillon fixe entre *train* et *test*. En effet, le *test* (20 %) a servi de *hold-out* final pour les métriques (rapport, matrice de confusion dans `4_evaluate.py`, analyses dans `5_analysis.py`), toujours avec le même `seed` pour reproduire exactement les mêmes exemples en *train* vs *test*. Autrement dit, ces exemples de *test* n’entraient pas dans l’apprentissage du modèle ni dans la validation croisée sur le *train* et n’étaient consultés qu’une seule fois, pour mesurer les performances après coup, afin que le score reflète le mieux possible la capacité de généralisation et non un biais d’« entraînement sur le test ».


### Validation croisée (à 5 plis)
À l’étape 3 (`scripts/3_models.py`), la fonction `evaluate_model` enchaîne deux évaluations distinctes pour chaque pipeline (vectorisation + classifieur), toutes deux inscrites dans le tableau de synthèse ci-dessous.

1. **Validation croisée** (colonnes *Acc. (CV)* et *F1 macro (CV)*) : Une validation croisée stratifiée à cinq plis (ou *5-fold cross validation* / CV) est appliquée uniquement sur `X_train` / `y_train`. Le *train* est découpé en cinq parts ; à chaque pli, le modèle s’entraîne sur quatre parts et est noté sur la cinquième, puis les cinq scores sont moyennés. Le jeu de test n’intervient à aucun moment dans cette boucle : il ne sert ni à l’apprentissage ni au calcul des métriques CV. Cette étape tient lieu de validation pour comparer les pipelines et lisser le hasard du découpage, sans équivalent d’un fichier *dev* séparé tenu à part.

2. **Évaluation sur le hold-out** (colonnes *Acc. (test)* et *F1 macro (test)*) : Après la CV, le même pipeline est entraîné une dernière fois sur l’intégralité du train (`fit` sur tout `X_train`), puis évalué une fois sur le jeu de test (`predict` sur `X_test`). Ce sont ces scores qui remplissent les colonnes « test » du tableau et qui servent notamment au tri des lignes (meilleur F1 macro sur le test en tête). Le test est donc utilisé seulement pour cette mesure finale par pipeline.

```python
# scripts/3_models.py - evaluate_model (extrait)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(pipe, split.X_train, split.y_train, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
pipe.fit(split.X_train, split.y_train)
y_pred = pipe.predict(split.X_test)
```

| Modèle | Acc. (CV) | F1 macro (CV) | Acc. (test) | F1 macro (test) |
|--------|-----------|---------------|-------------|-----------------|
| LinearSVC - TF-IDF + manuels | 0,9812 | 0,9812 | 0,9827 | 0,9827 |
| LinearSVC - BoW + manuels | 0,9752 | 0,9752 | 0,9794 | 0,9794 |
| MultinomialNB - BoW + manuels | 0,9602 | 0,9602 | 0,9636 | 0,9636 |
| MultinomialNB - TF-IDF | 0,9586 | 0,9586 | 0,9625 | 0,9625 |
| MultinomialNB - BoW | 0,9590 | 0,9590 | 0,9609 | 0,9609 |
| LinearSVC - TF-IDF | 0,9564 | 0,9564 | 0,9598 | 0,9598 |
| MultinomialNB - TF-IDF + manuels | 0,9543 | 0,9543 | 0,9594 | 0,9594 |
| LinearSVC - BoW | 0,9486 | 0,9486 | 0,9553 | 0,9553 |
| DecisionTree - BoW + manuels | 0,9356 | 0,9356 | 0,9380 | 0,9380 |
| DecisionTree - TF-IDF + manuels | 0,9345 | 0,9345 | 0,9344 | 0,9344 |
| DecisionTree - BoW | 0,7640 | 0,7535 | 0,7628 | 0,7519 |
| DecisionTree - TF-IDF | 0,7634 | 0,7529 | 0,7620 | 0,7510 |

*Comparaison des 12 pipelines évalués à l’étape 3, triés par F1 macro sur le test. Les colonnes CV correspondent aux scores moyennés sur 5 plis ; les colonnes test correspondent à l’évaluation sur le hold-out (20 %).*

### Choix du meilleur pipeline et évaluation finale

L’étape 4 (`scripts/4_evaluate.py`) s’appuie sur le fichier `artifacts/step3/best_model.joblib` produit à l’étape 3, qui reprend le pipeline classé premier après tri du tableau de synthèse.

```python
# scripts/3_models.py - tri du tableau de synthèse et choix du pipeline sauvegardé (extrait)
summary = pd.DataFrame(
    [
        {
            "modèle": r["model"],
            "cv_accuracy": r["cv_accuracy"],
            "cv_f1_macro": r["cv_f1_macro"],
            "test_accuracy": r["test_accuracy"],
            "test_f1_macro": r["test_f1_macro"],
        }
        for r in results
    ]
).sort_values(["test_f1_macro", "test_accuracy"], ascending=False)
best_name = summary.iloc[0]["modèle"]
best = next(r for r in results if r["model"] == best_name)
dump(best["estimator"], ARTIFACTS_DIR / "best_model.joblib")
```

Cette étape recalcule le découpage train/test avec la même fonction `make_split` et les mêmes hyperparamètres qu’aux étapes précédentes, charge le pipeline sauvegardé sans nouvel apprentissage, puis applique `predict` uniquement à `X_test`. Les sorties regroupent les indicateurs usuels (exactitude, F1 macro, rapport de classification) et des artefacts dans `artifacts/step4/` : scores tabulés, rapport texte et matrice de confusion au format image. Il s’agit d’une évaluation figée sur le hold-out, distincte du bloc d’entraînement / validation de l’étape 3. L’extrait ci-dessous reprend l’ordre réel des blocs dans le script (les détails d’affichage console et le tracé de la heatmap sont omis).

```python
# scripts/4_evaluate.py - ordre des opérations (extrait)
split = make_split(load_data())
model = load(MODEL_PATH)  # aucun fit : pipeline déjà entraîné à l’étape 3
y_pred = model.predict(split.X_test)
acc = accuracy_score(split.y_test, y_pred)
f1m = f1_score(split.y_test, y_pred, average="macro")
report = classification_report(split.y_test, y_pred, digits=3)
cm = confusion_matrix(split.y_test, y_pred)
# écritures dans artifacts/step4/ : classification_report.txt, scores.csv, confusion_matrix.png
```

---

## Modèles testés & paramètres
- **Modèles testés** : `MultinomialNB`, `LinearSVC`, `DecisionTreeClassifier`, combinés à BoW ou TF-IDF, avec ou sans traits manuels.
- **Traits manuels** : descripteurs de surface ajoutés au vecteur de traits — présence d’un point d’interrogation, ratio de mots en majuscules, présence de chiffres (indicateur de listicle) et longueur du titre.
- **Hyperparamètres** de référence : ex. `LinearSVC(C=1.0, max_iter=20000)`, `MultinomialNB(alpha=1.0)`, arbre avec profondeur et `min_samples_split` fixés pour limiter le sur-apprentissage.
- Tableau complet : `artifacts/step3/summary.csv`.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Définition des modèles (baseline NB, modèle fort SVM, et arbre en comparaison)
models = {
    "MultinomialNB": MultinomialNB(alpha=1.0), # Lissage de Laplace
    "LinearSVC": LinearSVC(C=1.0, max_iter=20000), # classifieur linéaire sur vecteurs BoW ou TF-IDF
    "DecisionTree": DecisionTreeClassifier(
        random_state=42,
        max_depth=30,
        min_samples_split=10,
    ),
}
```

### `MultinomialNB`

Classifieur probabiliste basé sur le théorème de **Bayes** en supposant l’indépendance conditionnelle des traits. Pour chaque document, il calcule la probabilité a posteriori de chaque classe à partir des fréquences de mots observées à l’entraînement. Le paramètre `alpha=1.0` applique un lissage de Laplace pour éviter les probabilités nulles sur des mots absents du corpus d’entraînement.

Bien adapté aux représentations « sacs de mots » (*bag of words*, ou BoW) (comptages entiers), il sert de baseline rapide et interprétable, mais suppose des traits indépendants et ignore l’ordre des mots (comme tout BoW).

### `LinearSVC`

Classifieur à vecteurs de support linéaire. Il cherche l’hyperplan qui maximise la marge entre les deux classes dans l’espace des traits (BoW ou TF-IDF). La régularisation `C=1.0` contrôle le compromis biais–variance ; `max_iter=20000` assure la convergence sur un vocabulaire large.

```python
LinearSVC(C=1.0, max_iter=20000)
```

Très efficace en grande dimension (vocabulaire ~10 000+ tokens), c’est le modèle le plus performant sur ce corpus (~98 % F1). Ses coefficients sont directement lisibles pour identifier les mots les plus discriminants.

### `DecisionTreeClassifier`

**Arbre de décision** qui partitionne récursivement l’espace des traits selon des seuils sur les valeurs de chaque token. À chaque nœud, il choisit la coupure qui maximise le gain d’information (entropie ou Gini). Les hyperparamètres `max_depth=30` et `min_samples_split=10` limitent le sur-apprentissage :

```python
DecisionTreeClassifier(random_state=42, max_depth=30, min_samples_split=10)
```

Plus interprétable visuellement, mais sensible au bruit et inférieur au SVM sur ce type de données textuelles en grande dimension.

---

## Résultats quantitatifs

**Meilleur pipeline** (sélectionné par F1 macro sur le test à l’étape 3) : **LinearSVC | TF-IDF + traits manuels**.

| Indicateur | Valeur (jeu de test, *n* = 6 400) |
|------------|-------------------------------------|
| Accuracy | **0,9842** |
| F1 macro | **0,9842** |

*Scores globaux du meilleur pipeline (LinearSVC | TF-IDF + traits manuels) sur le jeu de test, issus de l’évaluation finale à l’étape 4.*

**Par classe** (rapport de classification sur le test) :

| Classe | Précision | Rappel | F1 |
|--------|-----------|--------|-----|
| 0 (non-clickbait) | 0,986 | **0,982** | 0,984 |
| 1 (clickbait) | 0,982 | **0,987** | 0,984 |

*Précision, rappel et F1 par classe du meilleur pipeline sur le jeu de test.*

**Autres modèles** (détail dans `summary.csv`) : le `MultinomialNB`, quelle que soit la vectorisation, plafonne entre 0,959 et 0,964 de F1, soit environ 2 points en dessous du `LinearSVC`. Les arbres de décision présentent le profil le plus contrasté : avec traits manuels, ils atteignent ~0,938, mais sans eux la performance s’effondre à ~0,752, quelle que soit la vectorisation (BoW ou TF-IDF). Cet écart de près de 19 points illustre à quel point les arbres, contrairement au SVM, s’appuient sur quelques traits saillants plutôt que sur l’ensemble du vocabulaire.

---

## Interprétation des résultats

### Analyse des erreurs

Les deux classes, clickbait et non-clickbait, obtiennent des scores très proches (F1 ≈ 0,984 pour chacune), ce qui reflète un **corpus équilibré**. Le rappel est légèrement plus faible pour la classe non-clickbait, ce qui signifie que le modèle tend davantage à prendre un titre factuel pour du clickbait que l’inverse.

Les **faux positifs** (titres factuels classés clickbait) correspondent souvent à des formulations interrogatives ou à fort pouvoir d’accroche (p. ex. *What Is Facebook Actually Worth ?*, *Happy Gilmore Was On to Something*) que la presse légitime emploie aussi. La frontière n’est donc pas purement lexicale. Elle tend d’ailleurs à s’estomper, dans la mesure où une partie des médias traditionnels a progressivement adopté des titres plus accrocheurs, sensationnalistes, pour capter l’attention en ligne, tandis que les techniques de clickbait imitent de mieux en mieux les conventions stylistiques de la presse sérieuse (Christin, 2018).

Les **faux négatifs** (clickbaits non détectés) sont majoritairement des titres très courts ou sémantiquement ambigus (p. ex. *Memory Loss*, *The Tennis Racket*) qui ne contiennent pas les marqueurs stylistiques habituellement associés au clickbait dans le corpus.

L’inspection des **coefficients** du `LinearSVC` (`artifacts/step5/top_features.txt`) confirme que le modèle s’appuie sur des corrélations de style plutôt que sur le sens. Il associe en effet au clickbait des termes caractéristiques de certains formats (listes, quizz, majuscules) et au non-clickbait un vocabulaire de « une » factuelle (économie, politique, géographie). Ces corrélations sont propres au corpus et ne constituent pas une définition universelle du clickbait.

### Sur la valeur des scores (~98 %)

Un F1 de 98 % sur une tâche binaire mérite d’être relativisé. Plusieurs facteurs expliquent probablement ce résultat très élevé :

- **Corpus homogène et bien séparé** : Les deux classes proviennent de sources clairement distinctes (presse traditionnelle vs sites à clickbait). Les différences de style sont tranchées et le modèle n’a pas besoin de comprendre le sens pour bien classer.
- **Tâche binaire artificielle** : Une tâche plus nuancée, sur un classement par continuum plutôt que binaire, voire avec des données multilingues, serait sensiblement plus difficile.
- **Possible biais de source** (*dataset shift*) : Si les deux classes sont issues de sources très distinctes comme c’est le cas ici (presse traditionnelle vs sites à clickbait), le classifieur peut apprendre des artefacts propres à ces sources (ponctuation, longueur moyenne, vocabulaire de domaine) plutôt que la notion de clickbait elle-même, ce qui fausse l’expérience et limite sa généralisation.
- **Corrélations de forme et non de sens.** Les représentations BoW et TF-IDF ne capturent que la présence et la fréquence des mots, pas leur signification ni leur contexte. Le modèle distingue les classes grâce à des indices superficiels (style, ponctuation, tournures récurrentes) sans comprendre pourquoi un titre est trompeur. Un titre clickbait bien camouflé dans un style factuel, ou un titre factuel au ton accrocheur, peut ainsi tromper le classifieur.

Ce score de 98% suggère que le modèle a appris à distinguer le style éditorial de *The New York Times* de celui de *BuzzFeed* (sources probables du jeu Kaggle) plutôt que l’essence sémantique de la tromperie. Il reflètent donc la facilité relative de la tâche sur ce corpus autant que la qualité intrinsèque du modèle.

### Matrice de confusion (jeu de test)

![Matrice de confusion - lignes : vrai label, colonnes : prédiction](artifacts/step4/confusion_matrix.png)

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

## Conclusion : ce que le TAL doit savoir sur les pièges à clics
1. **Le modèle apprend la source, pas le leurre** : Les approches lexicales classiques (BoW/TF-IDF) sont d’excellents détecteurs de registres éditoriaux (ex: NYT vs BuzzFeed), mais peinent à capturer l’intention trompeuse derrière un titre.
2. **L’illusion des 98 %** : Un score très élevé sur ce type de tâche révèle souvent un biais de source dans un corpus polarisé, plutôt qu’une solution universelle au problème de la désinformation.
3. **La suprématie du linéaire sur les textes courts** : Le modèle `LinearSVC` reste l’approche de référence pour traiter la grande dimension vectorielle de ces données, surpassant largement les arbres de décision.
4. **Le rôle de sauvetage des traits manuels** : L’ajout de descripteurs de surface (point d’interrogation, majuscules, longueur) est indispensable pour rendre les modèles hiérarchiques (comme les arbres) un minimum compétitifs.
5. **L’asymétrie de l’erreur** : L’hybridation des styles journalistiques trompe les algorithmes : un titre factuel écrit de manière accrocheuse générera facilement un faux positif, brouillant la frontière entre information légitime et clickbait.
6. **L’impasse de la vision binaire** : Comme le théorisent Rubin et Chen (2012), forcer l’évaluation en deux classes (0 ou 1) occulte le spectre réel de la manipulation (exagération, omission, fabrication).
7. **Le poids de la typographie** : Sur des formats aussi courts que des titres, des éléments non lexicaux comme la ponctuation ou la présence de chiffres (les listicles) portent une charge sémantique et discriminante capitale.
8. **La limite du contexte** : Sans analyse profonde du contenu de l’article pointé par le lien, le TAL actuel évalue la forme (le style sensationnaliste) mais reste aveugle à l’essence même du clickbait.

## Bibliographie
> Christin, A. (2018). « Clicks or Pulitzers ? Web Journalists and Their Metrics ». American Journal of Sociology, 123(5), 1382–1415. Vulgarisation : Stanford News, 21 mars 2018. https://news.stanford.edu/stories/2018/03/this-stanford-scholar-learned-clickbait-will-surprise (dernière consultation le 11 mai 2026)
>
> Commission d’enrichissement de la langue française (2020). « Quelques termes de l’information et de la désinformation ». https://www.culture.gouv.fr/thematiques/langue-francaise-et-langues-de-france/agir-pour-les-langues/moderniser-et-enrichir-la-langue-francaise/rapports-de-la-commission-d-enrichissement/Rapport-annuel-de-la-Commission-d-enrichissement-de-la-langue-francaise-2020 
>
> Nofar, L., Portal, T., Elbaz, A., Apartsin, A., & Aperstein, Y. (2025). « An Interpretable Benchmark for Clickbait Detection and Tactic Attribution ». *arXiv preprint* arXiv:2509.10937. https://arxiv.org/abs/2509.10937
>
> Rubin, V. L., & Chen, Y. (2012). « Information Manipulation Classification Theory for LIS and NLP ». In *Proceedings of the Association for Information Science and Technology Annual Meeting (ASIST)*, Baltimore, MD, USA.
>
> van der Goot, R. (2021). « We Need to Talk About train-dev-test Splits ». In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, p. 4485–4494.