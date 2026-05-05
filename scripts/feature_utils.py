"""
Fonctions partagées par plusieurs scripts du pipeline.

Pourquoi ce fichier ?
---------------------
Quand on sauvegarde un modèle scikit-learn avec `joblib`, certains objets (par exemple une
fonction utilisée dans `FunctionTransformer`) doivent être retrouvés au chargement.
Si la fonction est définie dans un script exécuté en tant que `__main__`, le chargement peut
échouer.

Solution : mettre les fonctions réutilisées dans un module importable (ce fichier), puis les
importer dans les scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# Noms des traits manuels (dans l'ordre exact de construction de la matrice)
MANUAL_FEATURE_NAMES: list[str] = [
    "manual::n_chars",
    "manual::n_words",
    "manual::has_number",
    "manual::has_ellipsis",
    "manual::n_exclaim",
    "manual::n_question",
    "manual::upper_ratio",
    "manual::starts_with_wh",
]


def manual_features(texts) -> sparse.csr_matrix:
    """
    Extrait quelques traits « faits main » à partir du texte brut.

    Entrée : liste/array/Series de strings.
    Sortie : matrice sparse (n_samples, n_features) compatible avec scikit-learn.
    """
    # Normalisation en Series de strings (robuste aux entrées list/array/Series)
    s = pd.Series(list(texts)).fillna("").astype(str)

    # Bloc 1 — Longueurs (tendances générales : clickbait souvent plus "dense" / formaté)
    n_chars = s.str.len().to_numpy(dtype=float)
    n_words = s.str.split().str.len().fillna(0).to_numpy(dtype=float)

    # Bloc 2 — Indices de style (ponctuation, chiffres, ellipses)
    has_number = s.str.contains(r"\d", regex=True).to_numpy(dtype=float)
    has_ellipsis = s.str.contains(r"\.\.\.|…", regex=True).to_numpy(dtype=float)
    n_exclaim = s.str.count("!").to_numpy(dtype=float)
    n_question = s.str.count(r"\?").to_numpy(dtype=float)

    # Bloc 3 — Indices de "mise en scène" (caps, questions type "wh-words")
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

