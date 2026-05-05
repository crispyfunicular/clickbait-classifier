#!/usr/bin/env python3
"""
Étape 5 - Analyse et commentaires (erreurs + traits discriminants).

Objectif
--------
Aller au-delà des scores et produire des éléments de discussion pour le compte-rendu :
- exemples de titres mal classés (faux positifs / faux négatifs)
- pour un modèle linéaire (ex: LinearSVC), inspection des traits les plus discriminants

Important
---------
Le modèle de l'étape 3 est une Pipeline scikit-learn. Pour extraire des poids (coef_),
il faut que le classifieur final expose `coef_` (modèles linéaires).

Usage (depuis la racine du dépôt) :
    python scripts/5_analysis.py
"""

from __future__ import annotations  # Permet d’utiliser des types dans les annotations sans les évaluer tout de suite

from dataclasses import dataclass
from pathlib import Path  # Manipulation des chemins

import warnings
import numpy as np  # Calculs numériques
import pandas as pd  # Manipulation de tableaux
from joblib import load  # Chargement du modèle sauvegardé
from sklearn.exceptions import ConvergenceWarning  # Warnings d'optimisation (LinearSVC)
from sklearn.model_selection import train_test_split  # Split reproductible

import feature_utils  # noqa: F401 (assure que manual_features est importable au chargement joblib)

DATA_PATH = Path(__file__).resolve().parent.parent / "clickbait_data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "step3" / "best_model.joblib"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "step5"


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
    """Split train/test identique aux étapes 2/3/4 (stratifié, random_state=42)."""
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


def get_feature_names_from_pipeline(model) -> list[str] | None:
    """
    Reconstruit des noms de features pour le cas typique du meilleur modèle (LinearSVC + TF-IDF + manuels).

    Cas géré :
    - Pipeline( steps=[("features", FeatureUnion([("text", vectorizer), ("manual", FunctionTransformer)])),
                      ("clf", LinearSVC)] )
    - Pipeline( steps=[("vect", vectorizer), ("clf", ...)] ) (pas de traits manuels)

    Si on ne sait pas reconstruire proprement les noms, on renvoie None.
    """
    steps = dict(model.named_steps)

    # Bloc 1 — Pipeline sans traits manuels : "vect" -> "clf"
    if "vect" in steps:
        vect = steps["vect"]
        if hasattr(vect, "get_feature_names_out"):
            return list(vect.get_feature_names_out())
        return None

    # Bloc 2 — Pipeline avec traits manuels : "features" -> "clf"
    if "features" not in steps:
        return None

    union = steps["features"]
    if not hasattr(union, "transformer_list"):
        return None

    names: list[str] = []
    for name, transformer in union.transformer_list:
        # Sous-bloc 2a — Features texte : CountVectorizer / TfidfVectorizer
        if hasattr(transformer, "get_feature_names_out"):
            txt_names = list(transformer.get_feature_names_out())
            names.extend([f"text::{t}" for t in txt_names])
            continue

        # Sous-bloc 2b — Features manuelles : on connaît la liste, dans l'ordre du script 3
        if name == "manual":
            names.extend(feature_utils.MANUAL_FEATURE_NAMES)
        else:
            # Transformer inconnu : mieux vaut abandonner plutôt que d'afficher des noms faux
            return None

    return names


def save_misclassified_examples(
    *,
    X_test: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    n: int = 12,
) -> None:
    """
    Sauvegarde des exemples d'erreurs, séparés en :
    - faux positifs : vrai 0, prédit 1
    - faux négatifs : vrai 1, prédit 0

    Si `scores` est disponible (decision_function), on trie par "confiance" (valeur absolue).
    """
    df = pd.DataFrame({"headline": X_test.values, "y_true": y_test, "y_pred": y_pred})
    if scores is not None:
        df = df.assign(score=scores)

    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()

    # Tri : en priorité les erreurs les plus "assumées" par le modèle (score élevé)
    if "score" in df.columns:
        fp = fp.reindex(fp["score"].abs().sort_values(ascending=False).index)
        fn = fn.reindex(fn["score"].abs().sort_values(ascending=False).index)

    fp.head(n).to_csv(ARTIFACTS_DIR / "false_positives.csv", index=False)
    fn.head(n).to_csv(ARTIFACTS_DIR / "false_negatives.csv", index=False)


def save_top_coefficients(model, feature_names: list[str], k: int = 25) -> None:
    """
    Sauvegarde les traits les plus discriminants si le classifieur final est linéaire.

    Convention : coef_ > 0 => pousse vers la classe 1 (clickbait)
                coef_ < 0 => pousse vers la classe 0 (non-clickbait)
    """
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        (ARTIFACTS_DIR / "top_features.txt").write_text(
            "Le classifieur final n'expose pas `coef_` (modèle non linéaire).\n",
            encoding="utf-8",
        )
        return

    coefs = np.asarray(clf.coef_).ravel()
    if len(coefs) != len(feature_names):
        (ARTIFACTS_DIR / "top_features.txt").write_text(
            "Impossible d'aligner les coefficients avec les noms de features.\n",
            encoding="utf-8",
        )
        return

    order_pos = np.argsort(coefs)[::-1][:k]
    order_neg = np.argsort(coefs)[:k]

    lines: list[str] = []
    lines.append("Traits les plus associés à la classe 1 (clickbait)\n")
    for i in order_pos:
        lines.append(f"{coefs[i]: .5f}\t{feature_names[i]}")
    lines.append("\nTraits les plus associés à la classe 0 (non-clickbait)\n")
    for i in order_neg:
        lines.append(f"{coefs[i]: .5f}\t{feature_names[i]}")
    lines.append("")

    (ARTIFACTS_DIR / "top_features.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Charge le meilleur modèle et produit des fichiers d'analyse pour le compte-rendu."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bloc 0 — Nettoyage de l'affichage : évite de polluer la sortie avec des warnings de convergence
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Bloc 1 — Données + split reproductible
    df = load_data()
    split = make_split(df)

    # Bloc 2 — Chargement du modèle de l'étape 3
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Modèle introuvable. Lance d'abord l'étape 3.\n"
            f"Chemin attendu : {MODEL_PATH}"
        )
    model = load(MODEL_PATH)

    # Bloc 3 — Prédictions + (si possible) score de décision pour trier les erreurs
    y_pred = model.predict(split.X_test)
    scores = None
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(split.X_test)
        except Exception:
            scores = None

    # Bloc 4 — Exemples d'erreurs (faux positifs / faux négatifs)
    save_misclassified_examples(
        X_test=split.X_test,
        y_test=split.y_test,
        y_pred=y_pred,
        scores=scores,
        n=12,
    )

    # Bloc 5 — Traits discriminants (si classifieur linéaire)
    # Pour récupérer des noms fiables, il faut que le vectoriseur ait été "fit".
    # Ici, on refit le modèle sur le train uniquement (c'est OK : on n'utilise pas le test pour fit).
    model.fit(split.X_train, split.y_train)
    feature_names = get_feature_names_from_pipeline(model)
    if feature_names is not None:
        save_top_coefficients(model, feature_names, k=25)
    else:
        (ARTIFACTS_DIR / "top_features.txt").write_text(
            "Impossible de reconstruire proprement les noms des features.\n",
            encoding="utf-8",
        )

    print("Sauvegardé dans :", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()

