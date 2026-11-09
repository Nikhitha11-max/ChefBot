"""Offline evaluation: per-user hold-out, rating-prediction error and top-N ranking metrics."""
from __future__ import annotations

import json
import os

import numpy as np

from .collaborative import ItemKNN, MatrixFactorization
from .content import ContentModel
from .data import load_data, ratings_matrix
from .hybrid import HybridRecommender, PopularityModel

K = 10
REL = 4  # a held-out rating >= 4 counts as "relevant"


def split(R, test_frac=0.2, seed=7):
    rng = np.random.default_rng(seed)
    train = R.copy()
    test = np.full_like(R, np.nan)
    for u in range(R.shape[0]):
        rated = np.where(~np.isnan(R[u]))[0]
        n_test = max(3, int(round(test_frac * len(rated))))
        t = rng.choice(rated, size=n_test, replace=False)
        test[u, t] = R[u, t]
        train[u, t] = np.nan
    return train, test


def dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def evaluate(seed=7, verbose=True):
    recipes, ratings = load_data()
    R = ratings_matrix(ratings, len(recipes))
    train, test = split(R, seed=seed)
    hyb = HybridRecommender(recipes, train)

    models = {
        "Random": None,
        "Popularity": hyb.pop,
        "Content-based": hyb.content,
        "Item k-NN": hyb.knn,
        "Matrix factorisation": hyb.mf,
        "Hybrid": "hybrid",
        "Hybrid + MMR (0.3)": "hybrid_mmr",
    }
    rng = np.random.default_rng(0)
    results = {}
    for name, model in models.items():
        P = R_ = 0.0
        nd = 0.0
        recommended = set()
        ild_sum = 0.0
        n_users = 0
        for u in range(train.shape[0]):
            tr = {int(i): float(train[u, i]) for i in np.where(~np.isnan(train[u]))[0]}
            te = {int(i): float(test[u, i]) for i in np.where(~np.isnan(test[u]))[0]}
            relevant = {i for i, r in te.items() if r >= REL}
            if not relevant:
                continue
            if name == "Random":
                sc = rng.random(len(recipes))
            elif model == "hybrid":
                sc = hyb.blend(tr)
            elif model == "hybrid_mmr":
                sc = None
            else:
                sc = model.score_user(tr)
            cand = [i for i in range(len(recipes)) if i not in tr]
            if model == "hybrid_mmr":
                top = [r.idx for r in hyb.recommend(tr, n=K, diversity=0.3)]
            else:
                top = sorted(cand, key=lambda i: -sc[i])[:K]
            hits = [1 if i in relevant else 0 for i in top]
            P += sum(hits) / K
            R_ += sum(hits) / len(relevant)
            ideal = dcg(sorted(hits, reverse=True) if False else [1] * min(len(relevant), K))
            nd += dcg(hits) / ideal
            recommended |= set(top)
            sims = [1 - hyb.content.sim[a, b] for ai, a in enumerate(top) for b in top[ai + 1:]]
            ild_sum += float(np.mean(sims))
            n_users += 1
        results[name] = {
            f"Precision@{K}": P / n_users, f"Recall@{K}": R_ / n_users, f"NDCG@{K}": nd / n_users,
            "Coverage": len(recommended) / len(recipes), "Diversity (ILD)": ild_sum / n_users,
        }

    # Rating-prediction error (only models that output ratings)
    errs = {}
    for name, model in {"Item k-NN": hyb.knn, "Matrix factorisation": hyb.mf, "Popularity": hyb.pop}.items():
        se, ae, n = 0.0, 0.0, 0
        for u in range(train.shape[0]):
            tr = {int(i): float(train[u, i]) for i in np.where(~np.isnan(train[u]))[0]}
            sc = model.score_user(tr)
            for i in np.where(~np.isnan(test[u]))[0]:
                e = sc[i] - test[u, i]
                se += e * e
                ae += abs(e)
                n += 1
        errs[name] = {"RMSE": float(np.sqrt(se / n)), "MAE": ae / n}
    out = {"ranking": results, "rating_error": errs,
           "n_users": int(R.shape[0]), "n_items": int(R.shape[1]),
           "n_ratings": int((~np.isnan(R)).sum()), "sparsity": float(np.isnan(R).mean())}
    if verbose:
        print(json.dumps(out, indent=2))
    return out


def tune_weights(tune_seed=7, check_seed=11):
    """Small grid search of blend weights on one split, re-checked on a different split."""
    recipes, ratings = load_data()
    R = ratings_matrix(ratings, len(recipes))
    grid = [(float(c), float(k), round(float(1 - c - k), 2)) for c in np.arange(0, 1.01, 0.1) for k in np.arange(0, 1.01 - c, 0.1)]
    grid = [(round(c, 1), round(k, 1), round(m, 1)) for c, k, m in grid if m >= -1e-9]

    def ndcg_for(seed, w):
        train, test = split(R, seed=seed)
        hyb = HybridRecommender(recipes, train, weights=w)
        tot, n = 0.0, 0
        for u in range(train.shape[0]):
            tr = {int(i): float(train[u, i]) for i in np.where(~np.isnan(train[u]))[0]}
            rel = {int(i) for i in np.where(test[u] >= REL)[0]}
            if not rel:
                continue
            sc = hyb.blend(tr)
            top = sorted([i for i in range(len(recipes)) if i not in tr], key=lambda i: -sc[i])[:K]
            tot += dcg([1 if i in rel else 0 for i in top]) / dcg([1] * min(len(rel), K))
            n += 1
        return tot / n

    scored = sorted(((ndcg_for(tune_seed, w), w) for w in grid), reverse=True)[:5]
    for s, w in scored:
        print(f"weights={w}  NDCG@10 tune={s:.3f}  check={ndcg_for(check_seed, w):.3f}")
    print("default (0.6,0.2,0.2):", round(ndcg_for(tune_seed, (0.6, 0.2, 0.2)), 3), round(ndcg_for(check_seed, (0.6, 0.2, 0.2)), 3))


if __name__ == "__main__":
    import sys
    if "--tune" in sys.argv:
        tune_weights()
        sys.exit()
    res = evaluate()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "results.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", path)


