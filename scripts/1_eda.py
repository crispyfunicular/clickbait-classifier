#!/usr/bin/env python3
"""
Étape 1 - Exploration des données (EDA).

Corpus : clickbait_data.csv (titres en anglais, label binaire clickbait / non-clickbait).

Usage (depuis la racine du dépôt) :
    python scripts/1_eda.py
"""

from __future__ import annotations  # Permet d’utiliser des types dans les annotations sans les évaluer tout de suite

from pathlib import Path  # Manipulation des chemins de fichiers

import matplotlib.pyplot as plt  # Création des graphiques
import numpy as np  # Opérations numériques sur les vecteurs de comptage
import pandas as pd  # Chargement et structuration des données tabulaires
import seaborn as sns  # Graphiques statistiques (histogrammes, thème)
from sklearn.feature_extraction.text import CountVectorizer  # Vectorisation texte (sac de mots, n-grammes)

DATA_PATH = Path(__file__).resolve().parent.parent / "clickbait_data.csv"


def load_data() -> pd.DataFrame:
    """Lit le CSV UTF-8 et normalise la colonne de label en int (0 / 1)."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["clickbait"] = df["clickbait"].astype(int)
    return df


def print_shape_and_missing(df: pd.DataFrame) -> None:
    """Affiche la taille du tableau, les types de colonnes et les valeurs manquantes."""
    print("Dimensions :", df.shape)
    print(df.dtypes)
    print("\nManquants par colonne :")
    print(df.isna().sum())
    print("\nLignes avec titre vide :", int(df["headline"].str.strip().eq("").sum()))


def top_terms(
    texts: pd.Series,
    *,
    max_features: int = 8000,
    n: int = 25,  # nombre de termes les plus fréquents à retourner (top-n)
) -> pd.DataFrame:
    """
    Compte les fréquences de mots (hors stopwords) et retourne les n plus fréquents.

    Paramètres
    ----------
    texts : séries de titres pour une seule classe.
    max_features : limite de taille du vocabulaire pour rester raisonnable en mémoire.
    n : nombre de lignes à retourner.
    """
    vectorizer = CountVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(sums)[::-1][:n]
    return pd.DataFrame({"terme": terms[order], "freq": sums[order]})


def top_bigrams(
    texts: pd.Series,
    *,
    n: int = 15,  # nombre de bigrammes les plus fréquents à retourner
) -> pd.DataFrame:
    """
    Comme top_terms, mais pour les bigrammes (deux mots consécutifs).
    Utile pour repérer des formulations typiques (questions, listes, etc.).
    """
    vectorizer = CountVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(2, 2),
        max_features=15000,
    )
    X = vectorizer.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(sums)[::-1][:n]
    return pd.DataFrame({"bigramme": terms[order], "freq": sums[order]})


def main():
    """Enchaîne chargement, statistiques descriptives, graphiques et extractions lexicales."""
    sns.set_theme(style="whitegrid")

    df = load_data()

    print("Aperçu des 5 premières lignes")
    print()
    print(df.head().to_string())
    print()

    print("Forme du jeu de données et valeurs manquantes")
    print()
    print_shape_and_missing(df)

    print()
    print("***")
    print("Distribution des classes (0 = authentique, 1 = clickbait)")
    print()
    counts = df["clickbait"].value_counts().sort_index()
    print(counts.to_string())

    # Barres : effectifs par classe
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=sns.color_palette("Set2", n_colors=len(counts)),
    )
    ax.set_xlabel("clickbait (0 = non, 1 = oui)")
    ax.set_ylabel("Nombre de titres")
    ax.set_title("Répartition des classes")
    plt.tight_layout()
    plt.show()

    # Mesures de longueur par titre (pour comparer les deux classes)
    df = df.assign(
        n_chars=df["headline"].str.len(),  # nombre de caractères (ponctuation et espaces inclus)
        n_words=df["headline"].str.split().str.len(),  # segments séparés par des espaces (~nombre de « mots », approximation)
    )

    print()
    print("***")
    print("Longueurs (caractères et mots) par classe")
    print()
    print(df.groupby("clickbait")[["n_chars", "n_words"]].describe().round(2).to_string())

    # Distributions par classe (superposition des histogrammes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(
        axes,
        ["n_chars", "n_words"],
        ["Nombre de caractères", "Nombre de mots"],
    ):
        sns.histplot(
            data=df,
            x=col,
            hue="clickbait",
            bins=40,
            kde=True,
            ax=ax,
            palette="Set2",
            multiple="layer",
        )
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    texts_cb = df.loc[df["clickbait"] == 1, "headline"]
    texts_ok = df.loc[df["clickbait"] == 0, "headline"]

    print()
    print("***")
    print("Mots les plus fréquents (sans les stopwords)")
    print()
    top_cb = top_terms(texts_cb, n=25)
    top_ok = top_terms(texts_ok, n=25)
    print("Clickbait - 25 premiers termes")
    print(top_cb.to_string(index=False))
    print("\nNon-clickbait - 25 premiers termes")
    print(top_ok.to_string(index=False))

    print()
    print("***")
    print("Bigrammes fréquents")
    print("Clickbait - top bigrammes")
    print(top_bigrams(texts_cb).to_string(index=False))
    print("\nNon-clickbait - top bigrammes")
    print(top_bigrams(texts_ok).to_string(index=False))

    # Tirage reproductible pour illustrer le corpus
    sample_size = 8
    rng = 42
    ex_ok = df.loc[df["clickbait"] == 0, "headline"].sample(sample_size, random_state=rng)
    ex_cb = df.loc[df["clickbait"] == 1, "headline"].sample(sample_size, random_state=rng)

    print()
    print("***")
    print("Exemples aléatoires (random_state=42)")
    print("Exemples - non-clickbait (0)")
    for i, t in enumerate(ex_ok, 1):
        print(f"{i}. {t}")
    print()
    print("Exemples - clickbait (1)")
    for i, t in enumerate(ex_cb, 1):
        print(f"{i}. {t}")


if __name__ == "__main__":
    main()
