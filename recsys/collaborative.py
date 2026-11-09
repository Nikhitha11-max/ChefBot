"""Collaborative filtering: item-based k-NN and biased matrix factorisation (ALS)."""
from __future__ import annotations

import numpy as np


class ItemKNN:
    """Item-item CF with adjusted-cosine similarity (Sarwar et al., 2001)."""

    def __init__(self, k: int = 20, shrink: float = 5.0):
        self.k, self.shrink = k, shrink

    def fit(self, R: np.ndarray):
        mask = ~np.isnan(R)
        user_mean = np.nanmean(np.where(mask, R, np.nan), axis=1, keepdims=True)
        C = np.where(mask, R - user_mean, 0.0)               # centred, missing -> 0
        num = C.T @ C
        norms = np.sqrt((C ** 2).sum(axis=0))
        den = np.outer(norms, norms)
        co = mask.T.astype(float) @ mask.astype(float)       # co-rating counts
        sim = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        self.sim = sim * (co / (co + self.shrink))           # shrink low-support similarities
        np.fill_diagonal(self.sim, 0.0)
        self.item_mean = np.nanmean(R, axis=0)
        self.global_mean = np.nanmean(R)
        self.item_mean = np.where(np.isnan(self.item_mean), self.global_mean, self.item_mean)
        self.item_counts = mask.sum(axis=0)
        return self

    def score_user(self, user_ratings: dict[int, float]) -> np.ndarray:
        n = len(self.item_mean)
        if not user_ratings:
            return np.full(n, self.global_mean)
        ids = np.array(list(user_ratings))
        r = np.array([user_ratings[i] for i in ids], dtype=float)
        base = r.mean() if r.std() > 1e-9 else 3.0            # all-equal ratings carry no spread
        centred = r - base
        scores = np.full(n, base)
        for j in range(n):
            s = self.sim[j, ids]
            top = np.argsort(-np.abs(s))[: self.k]
            denom = np.abs(s[top]).sum()
            if denom > 1e-9:
                scores[j] = base + (s[top] * centred[top]).sum() / denom
        return np.clip(scores, 1, 5)


class MatrixFactorization:
    """Biased MF trained with alternating least squares; new users are 'folded in' with ridge."""

    def __init__(self, factors: int = 8, reg: float = 6.0, iters: int = 15, seed: int = 0):
        self.f, self.reg, self.iters, self.seed = factors, reg, iters, seed

    def fit(self, R: np.ndarray):
        rng = np.random.default_rng(self.seed)
        mask = ~np.isnan(R)
        nu, ni = R.shape
        self.mu = np.nanmean(R)
        cnt_i = mask.sum(0)
        self.bi = np.where(cnt_i > 0, (np.where(mask, R - self.mu, 0).sum(0)) / (cnt_i + 5), 0.0)
        cnt_u = mask.sum(1)
        bu = (np.where(mask, R - self.mu - self.bi, 0).sum(1)) / (cnt_u + 5)
        E = np.where(mask, R - self.mu - self.bi - bu[:, None], 0.0)
        P = rng.normal(0, 0.1, (nu, self.f))
        Q = rng.normal(0, 0.1, (ni, self.f))
        I = np.eye(self.f)
        for _ in range(self.iters):
            for u in range(nu):
                m = mask[u]
                if m.any():
                    Qm = Q[m]
                    P[u] = np.linalg.solve(Qm.T @ Qm + self.reg * I, Qm.T @ E[u, m])
            for i in range(ni):
                m = mask[:, i]
                if m.any():
                    Pm = P[m]
                    Q[i] = np.linalg.solve(Pm.T @ Pm + self.reg * I, Pm.T @ E[m, i])
        self.Q = Q
        return self

    def score_user(self, user_ratings: dict[int, float]) -> np.ndarray:
        """Fold-in: estimate user bias and factors from this user's ratings only."""
        n = len(self.bi)
        if not user_ratings:
            return np.full(n, self.mu) + self.bi
        ids = np.array(list(user_ratings))
        r = np.array([user_ratings[i] for i in ids], dtype=float)
        resid = r - self.mu - self.bi[ids]
        bu = resid.sum() / (len(ids) + 5)
        Qm = self.Q[ids]
        p = np.linalg.solve(Qm.T @ Qm + self.reg * np.eye(self.f), Qm.T @ (resid - bu))
        return np.clip(self.mu + bu + self.bi + self.Q @ p, 1, 5)
