"""Weighted hybrid recommender with cold-start handling, MMR diversity and explanations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .collaborative import ItemKNN, MatrixFactorization
from .content import ContentModel
from .data import Catalog, load_data, ratings_matrix


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x, dtype=float) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


@dataclass
class Recommendation:
    idx: int
    score: float
    reasons: list = field(default_factory=list)


class PopularityModel:
    """Bayesian (shrunk) mean rating - the cold-start fallback and an evaluation baseline."""

    def __init__(self, m: float = 5.0):
        self.m = m

    def fit(self, R):
        cnt = (~np.isnan(R)).sum(0)
        s = np.nansum(R, axis=0)
        g = np.nanmean(R)
        self.score = (s + self.m * g) / (cnt + self.m)
        self.count = cnt
        self.mean = np.where(cnt > 0, s / np.maximum(cnt, 1), g)
        return self

    def score_user(self, user_ratings):
        return self.score.copy()


class HybridRecommender:
    """Blend of content-based, item-kNN and matrix-factorisation scores.

    alpha = min(1, n_ratings / ramp) is the system's confidence that it knows the user.
    final = alpha * (w_c*content + w_k*knn + w_m*mf) + (1 - alpha) * popularity
    so brand-new users get popular items and the blend shifts to personalised scores as they rate.
    """

    def __init__(self, recipes, R, weights=(0.60, 0.20, 0.20), ramp: int = 5):
        self.recipes, self.catalog = recipes, Catalog(recipes)
        self.w, self.ramp = weights, ramp
        self.content = ContentModel(recipes)
        self.knn = ItemKNN().fit(R)
        self.mf = MatrixFactorization().fit(R)
        self.pop = PopularityModel().fit(R)

    @classmethod
    def from_data(cls, data_dir=None, **kw):
        recipes, ratings = load_data(data_dir) if data_dir else load_data()
        return cls(recipes, ratings_matrix(ratings, len(recipes)), **kw)

    # ---- scoring -----------------------------------------------------------------------
    def component_scores(self, user_ratings):
        return {
            "content": minmax(self.content.score_user(user_ratings)),
            "knn": minmax(self.knn.score_user(user_ratings)),
            "mf": minmax(self.mf.score_user(user_ratings)),
            "pop": minmax(self.pop.score_user(user_ratings)),
        }

    def blend(self, user_ratings, weights=None):
        c = self.component_scores(user_ratings)
        w = weights or self.w
        alpha = min(1.0, len(user_ratings) / self.ramp)
        personal = w[0] * c["content"] + w[1] * c["knn"] + w[2] * c["mf"]
        return alpha * personal + (1 - alpha) * c["pop"]

    # ---- recommendation ----------------------------------------------------------------
    def recommend(self, user_ratings, n=5, exclude=(), tag=None, diversity=0.0, weights=None, require=(), max_minutes=None):
        scores = self.blend(user_ratings, weights)
        banned = set(user_ratings) | set(exclude)
        cand = [i for i in range(len(scores))
                if i not in banned and (tag is None or tag in self.catalog.tags[i])
                and self.catalog.satisfies(i, require)
                and (max_minutes is None or self.recipes.minutes[i] <= max_minutes)]
        picked = self._mmr(scores, cand, n, diversity)
        return [Recommendation(i, float(scores[i]), self.explain(i, user_ratings)) for i in picked]

    def _mmr(self, scores, cand, n, diversity):
        """Maximal Marginal Relevance (Carbonell & Goldstein, 1998): trade relevance for variety."""
        lam = 1.0 - diversity
        chosen, pool = [], list(cand)
        while pool and len(chosen) < n:
            def mmr(i):
                red = max((self.content.sim[i, j] for j in chosen), default=0.0)
                return lam * scores[i] - (1 - lam) * red
            best = max(pool, key=mmr)
            chosen.append(best)
            pool.remove(best)
        return chosen

    def similar(self, item_idx, n=5, exclude=()):
        s = 0.5 * minmax(self.content.similar_to(item_idx)) + 0.5 * minmax(np.clip(self.knn.sim[item_idx], 0, None))
        s[item_idx] = -1
        order = [i for i in np.argsort(-s) if i not in set(exclude) and i != item_idx][:n]
        out = []
        for i in order:
            shared = sorted(self.catalog.tags[i] & self.catalog.tags[item_idx])
            reasons = [f"Similar to {self.catalog.short(item_idx)}" + (f" (shared: {', '.join(shared)})" if shared else "")]
            out.append(Recommendation(int(i), float(s[i]), reasons))
        return out

    # ---- explanations (transparency) ---------------------------------------------------
    def explain(self, i, user_ratings, max_reasons=2):
        reasons = []
        liked = [(j, r) for j, r in user_ratings.items() if r >= 4]
        if liked:
            j, r = max(liked, key=lambda jr: self.content.sim[i, jr[0]] * 1.0 + 0.01 * jr[1])
            shared = sorted(self.catalog.tags[i] & self.catalog.tags[j])
            if shared and self.content.sim[i, j] > 0.05:
                reasons.append(f"Because you rated {self.catalog.short(j)} {int(r)}/5 - shared: {', '.join(shared)}")
        if self.pop.count[i] >= 5:
            reasons.append(f"Rated {self.pop.mean[i]:.1f}/5 by {int(self.pop.count[i])} cooks")
        if not reasons:
            reasons.append("Popular with cooks overall")
        return reasons[:max_reasons]

    def favourite_tags(self, user_ratings, top=2, min_rating=4):
        counts = {}
        for i, r in user_ratings.items():
            if r >= min_rating:
                for g in self.catalog.tags[i]:
                    counts[g] = counts.get(g, 0) + 1
        return [g for g, c in sorted(counts.items(), key=lambda kv: -kv[1])[:top] if c >= 2]
