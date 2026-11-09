"""Content-based filtering: TF-IDF over tags + plot description, cosine similarity."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tag_tokens(tags: str) -> str:
    return " ".join("tag" + re.sub(r"[^a-z]", "", g.lower()) for g in tags.split("|"))


class ContentModel:
    """Represents each recipe as a TF-IDF vector; scores items against a user's taste profile."""

    def __init__(self, recipes, tag_weight: int = 3):
        # Repeating tag tokens up-weights them relative to plot words (modification #1).
        docs = [(_tag_tokens(g) + " ") * tag_weight + d for g, d in zip(recipes.tags, recipes.description)]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(docs)                 # L2-normalised rows
        self.sim = cosine_similarity(self.X)                  # item-item similarity (N x N)
        np.fill_diagonal(self.sim, 1.0)

    def score_user(self, user_ratings: dict[int, float]) -> np.ndarray:
        """Taste profile = rating-centred sum of liked/disliked item vectors; score = cosine."""
        n = self.X.shape[0]
        if not user_ratings:
            return np.zeros(n)
        profile = np.zeros(self.X.shape[1])
        for i, r in user_ratings.items():
            # Centre on the neutral midpoint (3), not the user's own mean: in a chat interface people
            # often say only "I loved X", and mean-centring would cancel an all-5 profile to zero.
            profile += (r - 3.0) * self.X[i].toarray().ravel()
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(n)
        return (self.X @ (profile / norm)).ravel()

    def similar_to(self, idx: int) -> np.ndarray:
        return self.sim[idx].copy()
