"""ChefBot - a conversational (and optionally proactive) front end for the hybrid recipe recommender.

Run:   python -m recsys.chatbot             (reactive: answers only when asked)
       python -m recsys.chatbot --proactive (also volunteers suggestions - can be switched off)
       python -m recsys.chatbot --demo      (plays a scripted conversation)

HCI design choices (see report): visible system status, user control (undo / reset / proactive
toggle), explanations on demand ("why"), persistent diet settings, and error tolerance (fuzzy
title matching).  NOTE: tags are not allergen information - this is not an allergy-safety tool.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re

from .data import norm
from .hybrid import HybridRecommender

HELP = """Here is what I understand:
  - "I loved Pad Thai and Chana Masala"     (also: liked / hated / disliked)
  - "rate Beef Stew 2"                       (rate a recipe 1-5)
  - "I'm vegetarian" / "I'm vegan" / "I'm gluten-free"   (diet filter kept for the whole chat)
  - "recommend" or "what should I cook?"     (add words: "something spicy", "quick dessert", "under 30 minutes")
  - "something like Butter Chicken"          (similar recipes)
  - "surprise me"  /  "more"                 (more variety / the next picks)
  - "why"                                    (explain the last recommendations)
  - "my ratings", "my diet", "undo", "reset" (see / fix your profile)
  - "proactive on|off"                       (let me volunteer suggestions, or not)
  - "quit"
Note: my tags are not allergen information - always check ingredients yourself.
"""

POS = r"(loved|love|liked|like|enjoyed|adored|really liked|really enjoyed)"
NEG = r"(hated|hate|disliked|didn't like|didnt like|did not like|can't stand|cant stand)"
DIETS = {"vegetarian": "Vegetarian", "vegan": "Vegan", "gluten free": "Gluten-Free", "gluten-free": "Gluten-Free"}


class ChatBot:
    def __init__(self, recommender: HybridRecommender | None = None, proactive: bool = False, profile_path: str | None = None):
        self.rec = recommender or HybridRecommender.from_data()
        self.cat = self.rec.catalog
        self.ratings: dict[int, float] = {}
        self.diet: set[str] = set()
        self.history: list[tuple[int, float | None]] = []      # for undo (idx, previous rating)
        self.last: list = []
        self.shown: set[int] = set()
        self.proactive = proactive
        self.profile_path = profile_path
        self.turns_since_nudge = 0
        self.nudged: set[int] = set()
        self.last_query: dict = {}
        if profile_path and os.path.exists(profile_path):
            with open(profile_path) as f:
                data = json.load(f)
            self.ratings = {int(k): float(v) for k, v in data.get("ratings", {}).items()}
            self.diet = set(data.get("diet", []))

    # ---- helpers -----------------------------------------------------------------------
    def save(self):
        if self.profile_path:
            with open(self.profile_path, "w") as f:
                json.dump({"ratings": self.ratings, "diet": sorted(self.diet)}, f)

    def _set_rating(self, idx, r):
        self.history.append((idx, self.ratings.get(idx)))
        self.ratings[idx] = float(r)
        self.save()

    def _format(self, recs, with_reasons=True):
        lines = []
        for n, r in enumerate(recs, 1):
            tags = ", ".join(sorted(self.cat.tags[r.idx]))
            lines.append(f"  {n}. {self.cat.title(r.idx)} [{tags}]")
            if with_reasons and r.reasons:
                lines.append(f"       -> {r.reasons[0]}")
        return "\n".join(lines)

    @staticmethod
    def _meal(now=None):
        h = (now or datetime.datetime.now()).hour
        return "breakfast" if h < 11 else "lunch" if h < 16 else "dinner"

    def greeting(self, now=None) -> str:
        if self.ratings:
            recs = self.rec.recommend(self.ratings, n=3, require=self.diet)
            self.last = recs
            msg = f"Welcome back! I remember {len(self.ratings)} of your ratings."
            if self.proactive:
                msg += f"\nIt's around {self._meal(now)} time. Ideas, without you asking:\n" + self._format(recs)
            return msg
        return ("Hi, I'm ChefBot. Tell me a dish you loved (e.g. \"I loved Pad Thai\"), tell me your diet "
                "(\"I'm vegetarian\"), or type \"help\".")

    def _ambiguous(self, phrase):
        close = self.cat.fuzzy(phrase)
        if close:
            names = " or ".join(f"\"{self.cat.short(i)}\"" for i in close)
            return f"I couldn't find \"{phrase.strip()}\" - did you mean {names}?"
        return f"I don't know a recipe called \"{phrase.strip()}\" yet. My catalog has {len(self.cat)} recipes; try another."

    # ---- main entry --------------------------------------------------------------------
    def respond(self, text: str) -> str:
        t = text.strip()
        low = t.lower()
        if not t:
            return "I'm listening - try \"help\" to see what I can do."
        reply = self._route(t, low)
        self.turns_since_nudge += 1
        quitting = low in {"quit", "exit", "bye"}
        nudge = self._nudge() if (self.proactive and not quitting) else ""
        return reply + (("\n\n" + nudge) if nudge else "")

    def _route(self, t, low):
        if low in {"quit", "exit", "bye"}:
            return "Bye! Your profile is " + ("saved." if self.profile_path else "not saved (use --profile).")
        if low in {"help", "?", "hi", "hello", "hey"}:
            return HELP
        if re.match(r"proactive\s+(on|off)", low):
            self.proactive = low.endswith("on")
            return ("OK - I'll occasionally suggest things on my own. Say \"proactive off\" any time."
                    if self.proactive else "Understood - I'll only answer when you ask.")
        if low in {"reset", "start over", "clear"}:
            self.ratings.clear(); self.history.clear(); self.shown.clear(); self.diet.clear(); self.save()
            return "Profile cleared (ratings and diet). We're starting fresh."
        if low == "undo":
            if not self.history:
                return "Nothing to undo."
            idx, prev = self.history.pop()
            if prev is None:
                self.ratings.pop(idx, None)
            else:
                self.ratings[idx] = prev
            self.save()
            return f"Undone: {self.cat.short(idx)}."
        if low in {"my ratings", "profile", "ratings"}:
            if not self.ratings:
                return "You haven't rated anything yet."
            rows = [f"  {self.cat.title(i)}: {int(r)}/5" for i, r in sorted(self.ratings.items(), key=lambda kv: -kv[1])]
            return "Your ratings:\n" + "\n".join(rows)
        if low in {"my diet", "diet"}:
            return ("Diet filter: " + ", ".join(sorted(self.diet)) + ". Say \"no diet\" to remove it.") if self.diet else "No diet filter set."
        if low in {"no diet", "clear diet"}:
            self.diet.clear(); self.save()
            return "Diet filter removed."
        m = re.match(r"(?:i am|i'm|im|i eat|i follow a|we are|we're)\s+(?:a\s+)?(vegetarian|vegan|gluten[- ]free)", low)
        if m:
            self.diet.add(DIETS[m.group(1).replace("  ", " ")])
            self.save()
            return (f"Got it - I'll only suggest {', '.join(sorted(self.diet))} recipes from now on. "
                    "(Tags are a guide, not an allergen guarantee.)")
        if low.startswith("why") or low.startswith("explain"):
            return self._why()

        m = re.match(r"rate\s+(.+?)\s+([1-5])(?:\s*/\s*5|\s*stars?)?$", low)
        if m:
            ids = self.cat.find_titles(m.group(1)) or self.cat.fuzzy(m.group(1), 1)
            if not ids:
                return self._ambiguous(m.group(1))
            self._set_rating(ids[0], int(m.group(2)))
            return f"Got it: {self.cat.short(ids[0])} = {m.group(2)}/5. {self._status()}"

        m = re.search(r"(?:something\s+like|similar\s+to|recipes?\s+like|dishes?\s+like|more\s+like)\s+(.+)", low)
        if m:
            ids = self.cat.find_titles(m.group(1)) or self.cat.fuzzy(m.group(1), 1)
            if not ids:
                return self._ambiguous(m.group(1))
            exclude = set(self.ratings) | {i for i in range(len(self.cat)) if not self.cat.satisfies(i, self.diet)}
            self.last = self.rec.similar(ids[0], n=5, exclude=exclude)
            self.shown |= {r.idx for r in self.last}
            return f"If you like {self.cat.short(ids[0])}, try:\n" + self._format(self.last)

        neg = re.search(NEG + r"\s+(.+)", low)
        pos = re.search(POS + r"\s+(.+)", low)
        if neg or pos:
            use_neg = neg is not None and (pos is None or neg.start() <= pos.start())
            m = neg if use_neg else pos
            phrase = m.group(2)
            ids = self.cat.find_titles(phrase)
            if not ids:
                if self.cat.tags_in(phrase):
                    return "I learn best from specific dishes - name a recipe you loved, or say \"recommend something " + phrase.strip() + "\"."
                return self._ambiguous(re.split(r"\s+and\s+|,", phrase)[0])
            val = 1 if use_neg else 5
            for i in ids:
                self._set_rating(i, val)
            names = " and ".join(self.cat.short(i) for i in ids)
            return f"Noted - {'you disliked' if use_neg else 'you loved'} {names}. {self._status()}"

        if low in {"more", "next", "more please"}:
            self.last = self.rec.recommend(self.ratings, n=5, exclude=self.shown, **self.last_query)
            self.shown |= {r.idx for r in self.last}
            return "More ideas:\n" + self._format(self.last) if self.last else "That's everything I have for that - try loosening a filter."

        if any(w in low for w in ("recommend", "suggest", "what should i cook", "what should i make", "surprise", "dinner", "lunch", "something", "ideas")):
            tags = self.cat.tags_in(low)
            mm = re.search(r"(?:under|less than|within|in|max(?:imum)?)\s+(\d+)\s*(?:min|minutes|mins)", low)
            max_minutes = int(mm.group(1)) if mm else None
            diversity = 0.6 if "surprise" in low or "different" in low or "variety" in low else 0.15
            require = set(self.diet) | {x for x in tags if x in {"Vegetarian", "Vegan", "Gluten-Free"}}
            tag = next((x for x in tags if x not in {"Vegetarian", "Vegan", "Gluten-Free"}), None)
            self.last_query = dict(tag=tag, require=require, max_minutes=max_minutes, diversity=diversity)
            self.last = self.rec.recommend(self.ratings, n=5, **self.last_query)
            self.shown = {r.idx for r in self.last}
            if not self.last:
                return "I have nothing that fits all of those filters - try relaxing one."
            bits = ([tag] if tag else []) + sorted(require) + ([f"under {max_minutes} min"] if max_minutes else [])
            head = "Here are my picks" + (f" ({', '.join(bits)})" if bits else "")
            if not self.ratings:
                head += " - crowd favourites; tell me what you like to personalise this"
            elif diversity > 0.3:
                head += " - mixing in some variety"
            return head + ":\n" + self._format(self.last) + "\nAsk \"why\" for explanations or \"more\" for others."

        ids = self.cat.find_titles(t)
        if ids:
            i = ids[0]
            return (f"{self.cat.title(i)} - {self.rec.recipes.description[i]}\n"
                    f"Say \"I loved {self.cat.short(i)}\" or \"rate {self.cat.short(i)} 3\" to teach me your taste.")
        return "Sorry, I didn't catch that. Try \"help\" for examples."

    def _status(self):
        n = len(self.ratings)
        alpha = min(1.0, n / self.rec.ramp)
        return f"(I know {n} of your tastes; personalisation {int(alpha * 100)}%.)"

    def _why(self):
        if not self.last:
            return "I haven't recommended anything yet - ask me to \"recommend\" first."
        lines = []
        for r in self.last:
            lines.append(f"  {self.cat.short(r.idx)}:")
            lines += [f"     - {x}" for x in self.rec.explain(r.idx, self.ratings, max_reasons=3)]
        return "Here's my reasoning:\n" + "\n".join(lines)

    # ---- proactive behaviour -----------------------------------------------------------
    def _nudge(self) -> str:
        """Volunteer a suggestion only when it is likely to be useful, and never too often."""
        if self.turns_since_nudge < 3 or len(self.ratings) < 3:
            return ""
        favs = self.rec.favourite_tags(self.ratings, top=1)
        if not favs:
            return ""
        recs = self.rec.recommend(self.ratings, n=1, tag=favs[0], exclude=self.shown | self.nudged, require=self.diet)
        if not recs:
            return ""
        self.turns_since_nudge = 0
        self.nudged.add(recs[0].idx)
        return (f"[Proactive tip] You keep rating {favs[0]} recipes highly - you might enjoy "
                f"{self.cat.title(recs[0].idx)}. ({recs[0].reasons[0]}). Say \"proactive off\" to stop these.")


DEMO = ["hello", "I'm vegetarian", "recommend", "I loved Chana Masala and Palak Paneer", "I hated Beef Stew",
        "rate Mapo Tofu 5", "recommend something quick under 30 minutes", "why", "something like Palak Paneer",
        "surprise me", "undo", "my ratings", "no diet", "quit"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="ChefBot recipe recommender")
    ap.add_argument("--proactive", action="store_true", help="allow the bot to volunteer suggestions")
    ap.add_argument("--profile", help="JSON file to remember your ratings and diet between sessions")
    ap.add_argument("--demo", action="store_true", help="run a scripted conversation")
    args = ap.parse_args(argv)
    bot = ChatBot(proactive=args.proactive, profile_path=args.profile)
    print("ChefBot:", bot.greeting())
    if args.demo:
        for line in DEMO:
            print(f"\nYou: {line}\nChefBot: {bot.respond(line)}")
        return
    while True:
        try:
            line = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print("ChefBot:", bot.respond(line))
        if norm(line) in {"quit", "exit", "bye"}:
            break


if __name__ == "__main__":
    main()
