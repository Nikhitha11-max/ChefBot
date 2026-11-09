"""Data loading and the Catalog helper (title lookup, tag lookup)."""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(data_dir: str = DATA_DIR):
    """Return (recipes_df, ratings_df). Recipe ids must be 1..N (contiguous)."""
    recipes = pd.read_csv(os.path.join(data_dir, "recipes.csv")).sort_values("recipe_id").reset_index(drop=True)
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    return recipes, ratings


def ratings_matrix(ratings: pd.DataFrame, n_items: int) -> np.ndarray:
    """Dense users x items matrix with NaN for unrated cells (item index = recipe_id - 1)."""
    users = sorted(ratings.user_id.unique())
    uidx = {u: i for i, u in enumerate(users)}
    R = np.full((len(users), n_items), np.nan)
    for u, m, r in ratings[["user_id", "recipe_id", "rating"]].itertuples(index=False):
        R[uidx[u], m - 1] = r
    return R


def norm(text: str) -> str:
    """Lower-case and strip punctuation so 'Pan's Labyrinth' matches 'pans labyrinth'."""
    text = text.lower().replace("&", " and ").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


TAG_ALIASES = {
    "vegetarian": "Vegetarian", "veggie": "Vegetarian", "vegan": "Vegan", "plant based": "Vegan",
    "gluten free": "Gluten-Free", "gluten-free": "Gluten-Free", "spicy": "Spicy", "hot": "Spicy",
    "healthy": "Healthy", "light": "Healthy", "comfort": "Comfort", "cozy": "Comfort", "hearty": "Comfort",
    "quick": "Quick", "fast": "Quick", "easy": "Quick",
    "dessert": "Dessert", "sweet": "Dessert", "baking": "Baking", "breakfast": "Breakfast", "brunch": "Breakfast",
    "soup": "Soup", "salad": "Salad", "pasta": "Pasta", "noodles": "Noodles", "curry": "Curry", "rice": "Rice",
    "seafood": "Seafood", "fish": "Seafood", "sandwich": "Sandwich", "grill": "Grill", "bbq": "Grill",
    "italian": "Italian", "mexican": "Mexican", "indian": "Indian", "thai": "Thai", "chinese": "Chinese",
    "japanese": "Japanese", "american": "American", "mediterranean": "Mediterranean", "french": "French",
    "middle eastern": "Middle-Eastern",
}


@dataclass
class Catalog:
    recipes: pd.DataFrame
    titles_norm: list = field(init=False)

    def __post_init__(self):
        self.titles_norm = [norm(t) for t in self.recipes.title]
        self.tags = [set(g.split("|")) for g in self.recipes.tags]

    def __len__(self):
        return len(self.recipes)

    def satisfies(self, idx: int, require) -> bool:
        """Diet logic: vegan dishes also satisfy 'Vegetarian'; other tags must match exactly."""
        have = self.tags[idx]
        for r in require:
            if r == "Vegetarian":
                if not ({"Vegetarian", "Vegan"} & have):
                    return False
            elif r not in have:
                return False
        return True

    def title(self, idx: int) -> str:
        return f"{self.recipes.title[idx]} ({self.recipes.minutes[idx]} min)"

    def short(self, idx: int) -> str:
        return self.recipes.title[idx]

    def find_titles(self, text: str) -> list[int]:
        """Find every catalog title mentioned in free text (longest match first, non-overlapping)."""
        t = f" {norm(text)} "
        found = []
        order = sorted(range(len(self)), key=lambda i: -len(self.titles_norm[i]))
        for i in order:
            name = self.titles_norm[i]
            variants = {name}
            if name.startswith("the "):
                variants.add(name[4:])
            if ":" in self.recipes.title[i]:                       # allow "pirates of the caribbean"
                variants.add(norm(self.recipes.title[i].split(":")[0]))
            for v in variants:
                pat = f" {v} "
                if len(v) >= 4 and pat in t:
                    found.append(i)
                    t = t.replace(pat, "  ", 1)
                    break
        return found

    def fuzzy(self, text: str, n: int = 3) -> list[int]:
        """Closest titles for a possibly misspelled phrase."""
        q = norm(text)
        names = self.titles_norm
        hits = difflib.get_close_matches(q, names, n=n, cutoff=0.6)
        return [names.index(h) for h in hits]

    def tags_in(self, text: str) -> list[str]:
        t = f" {norm(text)} "
        out = []
        for alias, g in TAG_ALIASES.items():
            if f" {norm(alias)} " in t and g not in out:
                out.append(g)
        return out
