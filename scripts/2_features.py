#!/usr/bin/env python3
"""
Étape 2 - Prétraitement et vectorisation.

Objectif
--------
Préparer des jeux de données prêts pour l'entraînement des modèles :
- séparation train/test stratifiée (pas de fuite de données)
- vectorisation Bag of Words (CountVectorizer)
- vectorisation TF-IDF (TfidfVectorizer)
- (optionnel) ajout de traits linguistiques simples (features manuelles)

Important : le vectoriseur est entraîné (fit) uniquement sur le train, puis appliqué (transform) sur le test.

Usage (depuis la racine du dépôt) :
    python scripts/2_features.py
"""

from __future__ import annotations  # Permet d’utiliser des types dans les annotations sans les évaluer tout de suite

from dataclasses import dataclass
from pathlib import Path  # Manipulation des chemins de fichiers

import numpy as np  # Opérations numériques sur les matrices de traits
import pandas as pd  # Chargement et structuration des données tabulaires
from joblib import dump  # Sérialisation des artefacts (vectoriseurs, matrices)
from scipy import sparse  # Matrices clairsemées (sparse) efficaces en mémoire
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Vectorisation texte (BoW, TF-IDF)
from sklearn.model_selection import train_test_split  # Découpage train/test stratifié

DATA_PATH = Path(__file__).resolve().parent.parent / "clickbait_data.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "step2"


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


def manual_features(texts: pd.Series) -> sparse.csr_matrix:
    """
    Extrait quelques traits « faits main » à partir du texte brut.

    Ces traits sont volontairement simples et interprétables. Ils sont calculés séparément sur train/test
    (pas de fit), puis concaténés aux matrices de vectorisation.
    """
    s = texts.fillna("").astype(str)

    n_chars = s.str.len().to_numpy(dtype=float)
    n_words = s.str.split().str.len().fillna(0).to_numpy(dtype=float)
    has_number = s.str.contains(r"\d", regex=True).to_numpy(dtype=float)
    has_ellipsis = s.str.contains(r"\.\.\.|…", regex=True).to_numpy(dtype=float)
    n_exclaim = s.str.count("!").to_numpy(dtype=float)
    n_question = s.str.count(r"\?").to_numpy(dtype=float)
    upper_ratio = (
        s.apply(lambda t: (sum(c.isupper() for c in t) / max(len(t), 1))).to_numpy(dtype=float)
    )
    starts_with_wh = (
        s.str.strip()
        .str.lower()
        .str.match(r"^(what|who|why|when|where|how)\b", na=False)
        .to_numpy(dtype=float)
    )

    feats = np.column_stack(
        [
            n_chars,
            n_words,
            has_number,
            has_ellipsis,
            n_exclaim,
            n_question,
            upper_ratio,
            starts_with_wh,
        ]
    )
    return sparse.csr_matrix(feats)


def fit_vectorizers(X_train: pd.Series) -> tuple[CountVectorizer, TfidfVectorizer]:
    """Crée et entraîne les vectoriseurs sur le train uniquement."""
    bow = CountVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
    )
    tfidf = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
    )
    bow.fit(X_train)
    tfidf.fit(X_train)
    return bow, tfidf


def vectorize(
    vectorizer,
    split: SplitData,
    *,
    add_manual: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Transforme train/test en matrices sparse, avec option de concaténation de traits manuels."""
    X_train_vec = vectorizer.transform(split.X_train)
    X_test_vec = vectorizer.transform(split.X_test)

    if add_manual:
        X_train_m = manual_features(split.X_train)
        X_test_m = manual_features(split.X_test)
        X_train_vec = sparse.hstack([X_train_vec, X_train_m], format="csr")
        X_test_vec = sparse.hstack([X_test_vec, X_test_m], format="csr")

    return X_train_vec, X_test_vec


def save_artifacts(
    *,
    split: SplitData,
    bow: CountVectorizer,
    tfidf: TfidfVectorizer,
    X_train_bow: sparse.csr_matrix,
    X_test_bow: sparse.csr_matrix,
    X_train_tfidf: sparse.csr_matrix,
    X_test_tfidf: sparse.csr_matrix,
    suffix: str,
) -> None:
    """Sauvegarde les matrices et les vectoriseurs pour réutilisation à l'étape 3."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dump(bow, ARTIFACTS_DIR / f"vectorizer_bow{suffix}.joblib")
    dump(tfidf, ARTIFACTS_DIR / f"vectorizer_tfidf{suffix}.joblib")

    dump(X_train_bow, ARTIFACTS_DIR / f"X_train_bow{suffix}.joblib")
    dump(X_test_bow, ARTIFACTS_DIR / f"X_test_bow{suffix}.joblib")
    dump(X_train_tfidf, ARTIFACTS_DIR / f"X_train_tfidf{suffix}.joblib")
    dump(X_test_tfidf, ARTIFACTS_DIR / f"X_test_tfidf{suffix}.joblib")

    dump(split.y_train, ARTIFACTS_DIR / "y_train.joblib")
    dump(split.y_test, ARTIFACTS_DIR / "y_test.joblib")


def main():
    """Charge les données, crée un split, vectorise (BoW et TF-IDF) et sauvegarde les artefacts."""
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

    bow, tfidf = fit_vectorizers(split.X_train)

    print()
    print("***")
    print("Vectorisation sans traits manuels")
    X_train_bow, X_test_bow = vectorize(bow, split, add_manual=False)
    X_train_tfidf, X_test_tfidf = vectorize(tfidf, split, add_manual=False)
    print("BoW   :", X_train_bow.shape, "->", X_test_bow.shape)
    print("TF-IDF:", X_train_tfidf.shape, "->", X_test_tfidf.shape)
    save_artifacts(
        split=split,
        bow=bow,
        tfidf=tfidf,
        X_train_bow=X_train_bow,
        X_test_bow=X_test_bow,
        X_train_tfidf=X_train_tfidf,
        X_test_tfidf=X_test_tfidf,
        suffix="",
    )
    print(f"Sauvegarde : {ARTIFACTS_DIR}")

    print()
    print("***")
    print("Vectorisation + traits manuels (optionnel)")
    X_train_bow_m, X_test_bow_m = vectorize(bow, split, add_manual=True)
    X_train_tfidf_m, X_test_tfidf_m = vectorize(tfidf, split, add_manual=True)
    print("BoW   + manuels:", X_train_bow_m.shape, "->", X_test_bow_m.shape)
    print("TF-IDF+ manuels:", X_train_tfidf_m.shape, "->", X_test_tfidf_m.shape)
    save_artifacts(
        split=split,
        bow=bow,
        tfidf=tfidf,
        X_train_bow=X_train_bow_m,
        X_test_bow=X_test_bow_m,
        X_train_tfidf=X_train_tfidf_m,
        X_test_tfidf=X_test_tfidf_m,
        suffix="_manual",
    )

    print()
    print("OK — Étape 2 terminée.")


if __name__ == "__main__":
    main()

