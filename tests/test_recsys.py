import numpy as np
import pytest

from recsys.chatbot import ChatBot
from recsys.data import Catalog, load_data, ratings_matrix
from recsys.hybrid import HybridRecommender


@pytest.fixture(scope="module")
def rec():
    return HybridRecommender.from_data()


def idx(rec, title):
    return list(rec.recipes.title).index(title)


def test_data_shapes():
    recipes, ratings = load_data()
    R = ratings_matrix(ratings, len(recipes))
    assert R.shape[1] == len(recipes) == 73
    assert ratings.rating.between(1, 5).all()
    assert (recipes.minutes > 0).all()


def test_title_matching():
    cat = Catalog(load_data()[0])
    assert len(cat.find_titles("i loved pad thai and chana masala")) == 2
    assert cat.short(cat.fuzzy("pad tai")[0]) == "Pad Thai"
    assert cat.tags_in("something spicy and quick") == ["Spicy", "Quick"] or set(cat.tags_in("something spicy and quick")) == {"Spicy", "Quick"}


def test_cold_start_returns_popular_items(rec):
    assert len(rec.recommend({}, n=5)) == 5


def test_excludes_rated_and_respects_filters(rec):
    ratings = {idx(rec, "Pad Thai"): 5.0}
    recs = rec.recommend(ratings, n=5, require={"Vegan"}, max_minutes=30)
    assert recs
    for r in recs:
        assert "Vegan" in rec.catalog.tags[r.idx]
        assert rec.recipes.minutes[r.idx] <= 30
        assert r.idx not in ratings


def test_vegan_satisfies_vegetarian(rec):
    cat = rec.catalog
    hummus = idx(rec, "Hummus and Pita")          # tagged Vegan only
    assert cat.satisfies(hummus, {"Vegetarian"})
    assert not cat.satisfies(idx(rec, "Beef Tacos"), {"Vegetarian"})


def test_personalisation_follows_taste(rec):
    liked = {idx(rec, t): 5.0 for t in ["Butter Chicken", "Chana Masala", "Lamb Rogan Josh", "Palak Paneer"]}
    top = rec.recommend(liked, n=10)
    # with only four ratings the blend is still noisy, so require a taste-related tag, not a perfect list
    assert sum(bool({"Indian", "Curry", "Spicy"} & rec.catalog.tags[r.idx]) for r in top) >= 4


def test_diversity_lowers_redundancy(rec):
    ratings = {idx(rec, t): 5.0 for t in ["Pad Thai", "Green Curry", "Tom Yum Soup"]}

    def ild(recs):
        ids = [r.idx for r in recs]
        return np.mean([1 - rec.content.sim[a, b] for i, a in enumerate(ids) for b in ids[i + 1:]])
    assert ild(rec.recommend(ratings, n=8, diversity=0.7)) >= ild(rec.recommend(ratings, n=8, diversity=0.0))


def test_explanations_present(rec):
    for r in rec.recommend({idx(rec, "Pad Thai"): 5.0, idx(rec, "Green Curry"): 5.0}, n=3):
        assert r.reasons


def test_chatbot_flow_and_undo():
    bot = ChatBot()
    assert "Noted" in bot.respond("I loved Pad Thai and Green Curry")
    assert len(bot.ratings) == 2
    assert "Undone" in bot.respond("undo")
    assert len(bot.ratings) == 1
    assert "did you mean" in bot.respond("I loved Pad Tai")
    assert "Here are my picks" in bot.respond("recommend")
    assert "reasoning" in bot.respond("why")
    assert "If you like Butter Chicken" in bot.respond("something like Butter Chicken")


def test_chatbot_diet_and_time_filters():
    bot = ChatBot()
    assert "vegetarian" in bot.respond("I'm vegetarian").lower()
    bot.respond("recommend something quick under 30 minutes")
    assert bot.last
    for r in bot.last:
        assert bot.cat.satisfies(r.idx, {"Vegetarian"})
        assert bot.rec.recipes.minutes[r.idx] <= 30
    assert "removed" in bot.respond("no diet")


def test_proactive_toggle():
    bot = ChatBot(proactive=True)
    bot.respond("I loved Pad Thai and Green Curry")
    bot.respond("rate Tom Yum Soup 5")
    replies = [bot.respond("hello") for _ in range(4)]
    assert any("Proactive tip" in r for r in replies)
    assert "only answer" in bot.respond("proactive off")
