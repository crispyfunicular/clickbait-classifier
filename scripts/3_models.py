#!/usr/bin/env python3
"""
Étape 3 - Mise en place des algorithmes de classification (et évaluation rapide).

Objectif
--------
Comparer plusieurs algorithmes imposés par les consignes en évitant toute fuite de données.
On utilise des Pipelines scikit-learn (vectorisation + classifieur) :
- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)

Modèles (au moins 2 parmi les 3 imposés) :
- Naive Bayes : MultinomialNB
- SVM linéaire : LinearSVC
- Arbre de décision : DecisionTreeClassifier

Usage (depuis la racine du dépôt) :
    python scripts/3_models.py
"""

from __future__ import annotations  # Permet d’utiliser des types dans les annotations sans les évaluer tout de suite

from dataclasses import dataclass
from pathlib import Path  # Manipulation des chemins de fichiers

import warnings
import numpy as np  # Opérations numériques et scores
import pandas as pd  # Chargement des données et tableau de synthèse
from joblib import dump  # Sauvegarde des modèles entraînés
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Vectorisation texte
from sklearn.exceptions import ConvergenceWarning  # Warnings d'optimisation (LinearSVC)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)  # Métriques d'évaluation
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)  # Split + CV
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes multinomial
from sklearn.pipeline import FeatureUnion, Pipeline  # Pipelines (vectoriseur + modèle) + union de features
from sklearn.preprocessing import FunctionTransformer  # Ajout de features manuelles
from sklearn.svm import LinearSVC  # SVM linéaire
from sklearn.tree import DecisionTreeClassifier  # Arbre de décision

from feature_utils import manual_features  # Features manuelles importables (joblib/pickle friendly)

DATA_PATH = Path(__file__).resolve().parent.parent / "clickbait_data.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "step3"


@dataclass(frozen=True)
class SplitData:
    """Conteneur des jeux train/test sous forme de séries pandas."""

    X_train: pd.Series
    X_test: pd.Series
    y_train: np.ndarray
    y_test: np.ndarray


def load_data() -> pd.DataFrame:
    """Lit le CSV UTF-8 et normalise la colonne de label en int (0 / 1)."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["clickbait"] = df["clickbait"].astype(int)
    return df


def make_split(df: pd.DataFrame) -> SplitData:
    """Sépare en train/test de manière stratifiée, avec random_state fixé pour la reproductibilité."""
    X = df["headline"].astype(str)
    y = df["clickbait"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def make_feature_pipeline(
    *,
    vectorizer,
    classifier,
    add_manual: bool,
) -> Pipeline:
    """
    Construit une Pipeline complète : features -> modèle.

    - Sans traits manuels : vectorizer -> classifier
    - Avec traits manuels : concat([vectorizer(text), manual(text)]) -> classifier
    """
    # Pipeline texte seule (vectorisation -> modèle)
    if not add_manual:
        return Pipeline([("vect", vectorizer), ("clf", classifier)])

    # Pipeline texte + traits manuels (concaténation -> modèle)
    features = FeatureUnion(
        transformer_list=[
            ("text", vectorizer),
            ("manual", FunctionTransformer(manual_features, validate=False)),
        ]
    )
    return Pipeline([("features", features), ("clf", classifier)])


def evaluate_model(
    name: str,
    pipe: Pipeline,
    split: SplitData,
    *,
    cv_folds: int = 5,
) -> dict:
    """Entraîne, calcule CV sur train, puis évalue sur test. Retourne un dict pour le tableau de synthèse."""
    # Validation croisée sur le train (mesure plus robuste que le score d'un seul split)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        pipe,
        split.X_train,
        split.y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    cv_f1 = cv_scores["test_f1_macro"].mean()
    cv_acc = cv_scores["test_accuracy"].mean()

    # Entraînement final sur tout le train, puis prédiction sur le test
    pipe.fit(split.X_train, split.y_train)
    y_pred = pipe.predict(split.X_test)

    # Évaluation sur le test (rapport + matrice de confusion)
    acc = accuracy_score(split.y_test, y_pred)
    f1m = f1_score(split.y_test, y_pred, average="macro")
    cm = confusion_matrix(split.y_test, y_pred)

    print()
    print("***")
    print(name)
    print("Confusion matrix (rows=true, cols=pred) :")
    print(cm)
    print()
    print(classification_report(split.y_test, y_pred, digits=3))

    return {
        "model": name,
        "cv_accuracy": float(cv_acc),
        "cv_f1_macro": float(cv_f1),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1m),
        "estimator": pipe,
    }


def main():
    """Charge les données, entraîne/évalue plusieurs pipelines et sauvegarde le meilleur modèle."""
    # Bloc 0 — Nettoyage de l'affichage : certains solveurs peuvent lever des warnings de convergence
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Chargement et split reproductible (point de départ commun à tous les modèles)
    df = load_data()
    split = make_split(df)

    print("Split train/test (stratifié, random_state=42)")
    print("Train :", len(split.X_train), "— Test :", len(split.X_test))
    print(
        "Distribution train (0/1) :",
        np.bincount(split.y_train).tolist(),
        "— test :",
        np.bincount(split.y_test).tolist(),
    )

    # Définition des vectorisations (mêmes hyperparamètres pour comparer "à armes égales")
    vectorizers = {
        "BoW": CountVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2),
        ),
        "TF-IDF": TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2),
        ),
    }

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

    # Boucle d'expériences (vectoriseur × modèle × avec/sans traits manuels)
    results: list[dict] = []
    for vec_name, vec in vectorizers.items():
        for model_name, clf in models.items():
            for add_manual in (False, True):
                suffix = " + manuels" if add_manual else ""
                name = f"{model_name} | {vec_name}{suffix}"
                pipe = make_feature_pipeline(vectorizer=vec, classifier=clf, add_manual=add_manual)
                results.append(evaluate_model(name, pipe, split))

    # Agrégation dans un tableau lisible + tri par performance sur le test
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

    print()
    print("***")
    print("Tableau de synthèse (trié par F1 macro sur test)")
    print(summary.round(4).to_string(index=False))

    # Sélection et sauvegarde du meilleur pipeline (réutilisable pour des analyses/étapes suivantes)
    best_name = summary.iloc[0]["modèle"]
    best = next(r for r in results if r["model"] == best_name)
    best_estimator = best["estimator"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_estimator, ARTIFACTS_DIR / "best_model.joblib")
    dump(summary, ARTIFACTS_DIR / "summary.joblib")
    summary.to_csv(ARTIFACTS_DIR / "summary.csv", index=False)

    print()
    print("Meilleur modèle (test F1 macro) :", best_name)
    print(f"Sauvegarde : {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

