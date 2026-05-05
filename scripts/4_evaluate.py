#!/usr/bin/env python3
"""
Étape 4 - Évaluation (métriques + matrice de confusion).

Objectif
--------
Produire une évaluation claire et reproductible du meilleur pipeline appris à l'étape 3.
On charge le modèle sauvegardé (`artifacts/step3/best_model.joblib`), on refait le split
train/test de manière déterministe (random_state=42), puis on évalue sur le test.

Ce script est volontairement « propre » :
- pas de ré-entraînement caché sur le test
- mêmes paramètres de split que dans les autres étapes
- sortie en console + fichiers dans `artifacts/step4/`

Usage (depuis la racine du dépôt) :
    python scripts/4_evaluate.py
"""

from __future__ import annotations  # Permet d’utiliser des types dans les annotations sans les évaluer tout de suite

from dataclasses import dataclass
from pathlib import Path  # Manipulation des chemins de fichiers

import numpy as np  # Opérations numériques
import pandas as pd  # Chargement et structuration des données tabulaires
import seaborn as sns  # Thèmes + heatmap
from joblib import load  # Chargement d'artefacts scikit-learn
from matplotlib import pyplot as plt  # Graphiques
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)  # Métriques
from sklearn.model_selection import train_test_split  # Découpage train/test stratifié

import feature_utils  # noqa: F401 (assure que manual_features est importable au chargement joblib)

DATA_PATH = Path(__file__).resolve().parent.parent / "clickbait_data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "step3" / "best_model.joblib"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "step4"


@dataclass(frozen=True)
class SplitData:
    """Conteneur des jeux train/test."""

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
    """Split train/test identique aux étapes 2 et 3 (stratifié, random_state=42)."""
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


def main():
    """Charge le meilleur modèle, calcule métriques et produit une matrice de confusion."""
    sns.set_theme(style="whitegrid")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bloc 1 — Données et split reproductible
    df = load_data()
    split = make_split(df)

    print("Split train/test (stratifié, random_state=42)")
    print("Train :", len(split.X_train), "— Test :", len(split.X_test))
    print(
        "Distribution test (0/1) :",
        np.bincount(split.y_test).tolist(),
    )

    # Bloc 2 — Chargement du modèle appris à l'étape 3
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Modèle introuvable. Lance d'abord l'étape 3.\n"
            f"Chemin attendu : {MODEL_PATH}"
        )
    model = load(MODEL_PATH)

    # Bloc 3 — Prédiction sur le test + métriques
    y_pred = model.predict(split.X_test)
    acc = accuracy_score(split.y_test, y_pred)
    f1m = f1_score(split.y_test, y_pred, average="macro")
    cm = confusion_matrix(split.y_test, y_pred)

    print()
    print("***")
    print("Scores sur le test")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1m:.4f}")
    print()
    report = classification_report(split.y_test, y_pred, digits=3)
    print(report)

    # Bloc 4 — Sauvegarde des résultats (texte + CSV + image)
    (ARTIFACTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    scores_df = pd.DataFrame(
        [
            {
                "accuracy": acc,
                "f1_macro": f1m,
                "n_test": int(len(split.y_test)),
            }
        ]
    )
    scores_df.to_csv(ARTIFACTS_DIR / "scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vrai label")
    ax.set_title("Matrice de confusion (test)")
    ax.set_xticklabels(["0 (non-clickbait)", "1 (clickbait)"], rotation=20, ha="right")
    ax.set_yticklabels(["0 (non-clickbait)", "1 (clickbait)"], rotation=0)
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)

    print()
    print("Sauvegardé dans :", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()

