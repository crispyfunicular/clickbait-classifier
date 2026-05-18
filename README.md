# clickbait-classifier

Projet de M1 pluriTAL — **Traitement statistique des données** : classification de titres d’articles en **clickbait** vs **non-clickbait** (NLP classique avec scikit-learn).

**Statut : projet rendu et livré** (mai 2026).

Dépôt : https://github.com/crispyfunicular/clickbait_classifier

---

## Résumé

- **Corpus** : 32 000 titres en anglais (`clickbait_data.csv`), classes équilibrées (0 = presse traditionnelle, 1 = sites à clickbait).
- **Pipeline** : EDA → vectorisation (BoW, TF-IDF) + traits manuels → comparaison de classifieurs → évaluation → analyse des erreurs.
- **Meilleur modèle (jeu de test)** : `LinearSVC` + TF-IDF + traits manuels — **F1 macro ≈ 0,984** (accuracy identique).
- **Compte-rendu** : [`rapport.md`](rapport.md) (source) ; PDF générable via Pandoc (voir ci-dessous).

Les scores élevés sont discutés dans le rapport (corpus homogène, frontière stylistique, corrélations de forme plutôt que de sens).

---

## Livrables

| Élément | Fichier / dossier |
|--------|-------------------|
| Données | `clickbait_data.csv` |
| Scripts | `scripts/1_eda.py` … `scripts/5_analysis.py`, `scripts/feature_utils.py` |
| Artefacts | `artifacts/step3/`, `step4/`, `step5/` |
| Rapport | `rapport.md` |
| Illustration | `clickbaits.png` |

---

## Structure du dépôt

```
clickbait-classifier/
├── clickbait_data.csv      # jeu de données
├── rapport.md              # compte-rendu (Markdown)
├── clickbaits.png
├── scripts/
│   ├── 1_eda.py            # exploration
│   ├── 2_features.py       # vectorisation, split train/test
│   ├── 3_models.py         # entraînement et comparaison des modèles
│   ├── 4_evaluate.py       # métriques, matrice de confusion
│   ├── 5_analysis.py       # faux positifs/négatifs, traits discriminants
│   └── feature_utils.py    # traits manuels, utilitaires partagés
└── artifacts/
    ├── step3/summary.csv
    ├── step4/              # scores, rapport, confusion_matrix.png
    └── step5/              # false_positives.csv, false_negatives.csv, top_features.txt
```

---

## Installation et exécution

Depuis la racine du dépôt :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ordre recommandé (chaque script écrit dans `artifacts/`) :

```bash
python scripts/1_eda.py
python scripts/2_features.py
python scripts/3_models.py
python scripts/4_evaluate.py
python scripts/5_analysis.py
```

**Reproductibilité** : split train/test 80 % / 20 %, stratifié, `random_state=42`. Les vectoriseurs sont ajustés (`fit`) uniquement sur le train.

---

## Modèles et résultats

Trois algorithmes imposés par la consigne, testés avec BoW ou TF-IDF, avec ou sans traits manuels :

| Modèle | Rôle |
|--------|------|
| `MultinomialNB` | baseline probabiliste |
| `LinearSVC` | meilleures performances |
| `DecisionTreeClassifier` | comparaison ; très dépendant des traits manuels |

**Résultats sur le jeu de test** (*n* = 6 400) — pipeline retenu : `LinearSVC | TF-IDF + manuels`

| Indicateur | Valeur |
|------------|--------|
| Accuracy | 0,9842 |
| F1 macro | 0,9842 |

| Classe | Précision | Rappel | F1 |
|--------|-----------|--------|-----|
| 0 (non-clickbait) | 0,986 | 0,982 | 0,984 |
| 1 (clickbait) | 0,982 | 0,987 | 0,984 |

Tableau complet des combinaisons modèle × vectorisation : `artifacts/step3/summary.csv`.

**Traits discriminants** (extraits, `LinearSVC`) :

- Plutôt **clickbait** : `manual::upper_ratio`, `text::guess`, `text::tell`, `text::things`…
- Plutôt **non-clickbait** : `text::wins`, `text::obama`, `text::economy`, `text::says`…

Convention : `text::token` = mot/n-gramme du vectoriseur ; `manual::trait` = descripteur calculé sur le titre brut.

---

## Générer le PDF du rapport

Avec [Pandoc](https://pandoc.org/) et XeLaTeX (caractères Unicode, français) :

```bash
pandoc rapport.md -o rapport.pdf --pdf-engine=xelatex \
  -V documentclass=article -V geometry:margin=2.5cm -V lang=fr \
  -V mainfont="DejaVu Serif"
```

Exécuter la commande depuis la racine du dépôt pour que les images (`artifacts/…`) soient résolues.

---

## Données

- **Fichier** : `clickbait_data.csv` (UTF-8)
- **Colonnes** : `headline`, `clickbait` (0 ou 1)
- **Source** : [Kaggle — Clickbait dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)
- **Non-clickbait** : *WikiNews*, *New York Times*, *The Guardian*, *The Hindu*
- **Clickbait** : *BuzzFeed*, *Upworthy*, *ViralNova*, etc.

---

## Références (rapport)

Rubin & Chen (2012) ; van der Goot (2021) ; Christin (2018) ; Nofar et al. (2025) ; Commission d’enrichissement de la langue française (2020). Détail complet dans [`rapport.md`](rapport.md#bibliographie).
