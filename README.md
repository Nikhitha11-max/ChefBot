# ChefBot - Hybrid Recipe Recommender with a Conversational, Proactive Interface

AI/HCI course project. A recipe recommendation system that combines **content-based filtering**
(TF-IDF + cosine similarity), **item-based collaborative filtering** and **matrix factorisation**,
used through a **chatbot** with persistent diet and time filters, explanations, and an optional
**proactive** mode.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m recsys.chatbot                  # reactive chatbot
python -m recsys.chatbot --proactive      # chatbot that volunteers suggestions (toggle: "proactive off")
python -m recsys.chatbot --profile me.json  # remember ratings + diet between sessions
python -m recsys.chatbot --demo --proactive # scripted conversation
python -m recsys.evaluate                 # offline evaluation -> docs/results.json
python -m recsys.evaluate --tune          # blend-weight grid search
python -m pytest -q                       # unit tests (11)
```

Example phrases: `I loved Pad Thai and Chana Masala`, `I'm vegetarian`, `recommend something quick under 30 minutes`,
`something like Butter Chicken`, `surprise me`, `why`, `more`, `undo`, `my ratings`, `no diet`, `reset`, `help`.

## Project layout
```
data/generate_data.py   builds recipes.csv (73 recipes) + ratings.csv (SYNTHETIC ratings, 150 simulated cooks)
recsys/data.py          loading, rating matrix, Catalog (title/tag lookup, diet logic, fuzzy matching)
recsys/content.py       TF-IDF content model (tag tokens up-weighted)
recsys/collaborative.py item k-NN (adjusted cosine) and ALS matrix factorisation with new-user fold-in
recsys/hybrid.py        weighted hybrid, cold-start ramp, hard filters, MMR diversity, explanations
recsys/evaluate.py      hold-out evaluation (RMSE/MAE, Precision/Recall/NDCG@10, coverage, diversity)
recsys/chatbot.py       ChefBot: intents, diet memory, undo/reset, "why", proactive nudges
tests/test_recsys.py    unit tests
docs/                   results.json, weight_tuning.txt, demo_transcript.txt
```



### What was changed or added relative to a basic content-based recommender
1. Hybrid of three models with a blend and a popularity fall-back (cold start).
2. Tag-token up-weighting in the TF-IDF documents.
3. New-user "fold-in" so a chat user gets recommendations without retraining.
4. Hard constraints: persistent diet filter (vegan satisfies vegetarian) and "under N minutes".
5. MMR diversity re-ranking ("surprise me").
6. Natural-language explanations ("Because you rated ...") and a "why" command.
7. Conversational interface with undo/reset, fuzzy title matching and an opt-in, rate-limited proactive mode.
8. Offline evaluation with baselines, plus unit tests.

## Using real data
Keep the same columns: `data/recipes.csv` (`recipe_id,title,minutes,tags,description`; ids 1..N, tags separated by `|`)
and `data/ratings.csv` (`user_id,recipe_id,rating` on a 1-5 scale). Public recipe-review datasets can be converted to this format.

## Limitations
- The bundled ratings are simulated from taste "archetypes", which favours content-based models; real-data results will differ.
- Tags are not allergen information. ChefBot is not an allergy-safety tool.
