"""Build the sample dataset used by the project.

recipes.csv - 73 recipes with cuisine/diet/style tags, cooking time and a short, original description.
ratings.csv - SYNTHETIC ratings for 150 simulated home cooks. Each simulated user has one or two
              "taste archetypes" (e.g. spice lover, healthy eater) plus a personal rating bias and
              noise. Synthetic data keeps the project self-contained and reproducible. To use real
              data (e.g. a public recipe-review dataset) keep the same columns - see README.

Run:  python data/generate_data.py
"""
import csv
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RECIPES = [
 ("Spaghetti Carbonara",25,"Italian|Pasta|Comfort|Quick","Spaghetti tossed with crisp pork, eggs and cheese into a silky peppery sauce."),
 ("Margherita Pizza",40,"Italian|Baking|Vegetarian|Comfort","Thin crust topped with tomato, fresh mozzarella and basil, baked until blistered."),
 ("Mushroom Risotto",45,"Italian|Rice|Vegetarian|Comfort|Gluten-Free","Creamy arborio rice stirred slowly with mushrooms, white wine and parmesan."),
 ("Lasagna Bolognese",90,"Italian|Pasta|Comfort|Baking","Layers of pasta, slow meat ragu, bechamel and cheese baked until bubbling."),
 ("Caprese Salad",10,"Italian|Salad|Vegetarian|Healthy|Quick|Gluten-Free","Sliced tomato, mozzarella and basil drizzled with olive oil and balsamic."),
 ("Tiramisu",30,"Italian|Dessert|Vegetarian","Coffee-soaked ladyfingers layered with whipped mascarpone cream and cocoa."),
 ("Minestrone Soup",50,"Italian|Soup|Vegan|Healthy|Comfort","Chunky vegetable and bean soup with small pasta in a tomato broth."),
 ("Chicken Parmesan",45,"Italian|Comfort|Baking","Breaded chicken cutlets baked under tomato sauce and melted cheese."),
 ("Beef Tacos",25,"Mexican|Quick|Spicy","Seasoned ground beef in warm tortillas with salsa, lettuce and cheese."),
 ("Chicken Enchiladas",50,"Mexican|Comfort|Baking|Spicy","Rolled tortillas filled with chicken and smothered in chili sauce and cheese."),
 ("Guacamole and Chips",10,"Mexican|Vegan|Quick|Healthy|Gluten-Free","Mashed avocado with lime, onion, cilantro and chili served with crisp chips."),
 ("Black Bean Burrito Bowl",25,"Mexican|Rice|Vegan|Healthy|Quick","Rice, black beans, corn and salsa piled in a bowl with lime and avocado."),
 ("Chilaquiles",30,"Mexican|Breakfast|Spicy|Comfort","Fried tortilla pieces simmered in red salsa and topped with egg and crema."),
 ("Fish Tacos",30,"Mexican|Seafood|Quick|Healthy","Grilled white fish in tortillas with cabbage slaw and lime crema."),
 ("Butter Chicken",50,"Indian|Curry|Comfort|Spicy|Gluten-Free","Tender chicken simmered in a creamy tomato and butter sauce with warm spices."),
 ("Chana Masala",35,"Indian|Curry|Vegan|Healthy|Spicy|Gluten-Free","Chickpeas stewed in a tangy tomato and onion gravy with ginger and garam masala."),
 ("Palak Paneer",40,"Indian|Curry|Vegetarian|Healthy|Gluten-Free","Cubes of fresh cheese in a smooth spiced spinach sauce."),
 ("Vegetable Biryani",60,"Indian|Rice|Vegetarian|Spicy|Comfort","Fragrant basmati rice layered with spiced vegetables, saffron and fried onion."),
 ("Lamb Rogan Josh",90,"Indian|Curry|Spicy|Comfort|Gluten-Free","Slow-cooked lamb in a deep red sauce of Kashmiri chili and yogurt."),
 ("Red Lentil Dal",30,"Indian|Soup|Vegan|Healthy|Quick|Gluten-Free","Soft red lentils simmered with turmeric, garlic and a sizzling cumin tempering."),
 ("Pad Thai",30,"Thai|Noodles|Quick|Spicy","Stir-fried rice noodles with egg, peanuts, lime and tamarind sauce."),
 ("Green Curry",40,"Thai|Curry|Spicy|Gluten-Free","Coconut curry with green chili paste, basil, bamboo shoots and chicken."),
 ("Tom Yum Soup",30,"Thai|Soup|Seafood|Spicy|Healthy|Gluten-Free","Hot and sour broth with shrimp, lemongrass, galangal and mushrooms."),
 ("Mango Sticky Rice",40,"Thai|Dessert|Vegan|Rice|Gluten-Free","Sweet coconut sticky rice served with slices of ripe mango."),
 ("Thai Basil Stir Fry",20,"Thai|Quick|Spicy|Rice","Minced meat and chilies wok-fried with holy basil and served over rice."),
 ("Kung Pao Chicken",30,"Chinese|Quick|Spicy","Diced chicken stir-fried with peanuts, dried chilies and a sweet-sour glaze."),
 ("Vegetable Fried Rice",20,"Chinese|Rice|Vegetarian|Quick","Day-old rice wok-tossed with egg, peas, carrots and soy sauce."),
 ("Mapo Tofu",30,"Chinese|Spicy|Vegetarian|Quick","Soft tofu in a numbing, chili-bean sauce with scallions."),
 ("Pork Dumplings",60,"Chinese|Comfort","Hand-folded dumplings filled with pork and cabbage, steamed or pan-fried."),
 ("Hot and Sour Soup",30,"Chinese|Soup|Spicy|Quick","Peppery vinegar broth with tofu, mushrooms and egg ribbons."),
 ("Salmon Sushi Bowl",25,"Japanese|Seafood|Rice|Healthy|Quick","Sushi rice topped with raw salmon, cucumber, avocado and sesame."),
 ("Chicken Katsu",35,"Japanese|Comfort|Rice","Panko-crusted fried chicken cutlet served with tangy brown sauce."),
 ("Miso Soup",15,"Japanese|Soup|Vegetarian|Healthy|Quick","Light dashi broth with miso paste, tofu and seaweed."),
 ("Vegetable Ramen",40,"Japanese|Noodles|Soup|Vegetarian|Comfort","Chewy noodles in a savory broth with mushrooms, corn and greens."),
 ("Teriyaki Salmon",25,"Japanese|Seafood|Quick|Healthy","Pan-seared salmon glazed with sweet soy teriyaki sauce."),
 ("Classic Cheeseburger",30,"American|Grill|Comfort|Sandwich","Juicy beef patty with melted cheese, pickles and a toasted bun."),
 ("Mac and Cheese",35,"American|Pasta|Vegetarian|Comfort|Baking","Elbow pasta in a rich cheddar sauce with a crunchy breadcrumb top."),
 ("Pancakes with Maple Syrup",25,"American|Breakfast|Dessert|Vegetarian|Quick","Fluffy buttermilk pancakes stacked high and drenched in maple syrup."),
 ("BBQ Pulled Pork Sandwich",240,"American|Grill|Sandwich|Comfort","Pork shoulder slow-cooked until tender, shredded in smoky barbecue sauce."),
 ("Chocolate Chip Cookies",35,"American|Dessert|Baking|Vegetarian","Soft, chewy cookies loaded with chocolate chunks and brown sugar."),
 ("Apple Pie",90,"American|Dessert|Baking|Vegetarian|Comfort","Cinnamon-spiced apples in a flaky double crust."),
 ("Clam Chowder",50,"American|Soup|Seafood|Comfort","Creamy New England soup with clams, potatoes and bacon."),
 ("Southern Fried Chicken",60,"American|Comfort","Buttermilk-brined chicken coated in seasoned flour and fried until crackly."),
 ("Greek Salad",15,"Mediterranean|Salad|Vegetarian|Healthy|Quick|Gluten-Free","Tomato, cucumber, olives and feta with oregano and olive oil."),
 ("Hummus and Pita",15,"Mediterranean|Vegan|Healthy|Quick","Smooth chickpea and tahini dip scooped up with warm pita bread."),
 ("Grilled Lemon Chicken",35,"Mediterranean|Grill|Healthy|Gluten-Free","Chicken marinated in lemon, garlic and herbs and grilled until charred."),
 ("Shakshuka",30,"Mediterranean|Breakfast|Vegetarian|Spicy|Gluten-Free","Eggs poached in a spiced tomato and pepper sauce, served with bread."),
 ("Falafel Wrap",35,"Middle-Eastern|Sandwich|Vegan|Healthy","Crisp chickpea fritters wrapped with pickles, greens and tahini sauce."),
 ("Lamb Kofta Skewers",35,"Middle-Eastern|Grill|Spicy|Gluten-Free","Spiced minced lamb shaped onto skewers and grilled with yogurt dip."),
 ("Baklava",75,"Middle-Eastern|Dessert|Baking|Vegetarian","Layers of crisp pastry, chopped nuts and honey syrup."),
 ("Tabbouleh",20,"Middle-Eastern|Salad|Vegan|Healthy|Quick","Finely chopped parsley with tomato, bulgur and lemon."),
 ("Seafood Paella",70,"Mediterranean|Rice|Seafood|Gluten-Free","Saffron rice cooked with shrimp, mussels and squid in a wide pan."),
 ("Ratatouille",60,"French|Vegan|Healthy|Gluten-Free","Slow-baked summer vegetables in a herby tomato base."),
 ("French Onion Soup",70,"French|Soup|Comfort|Vegetarian","Deeply caramelized onions in broth under a melted cheese crouton."),
 ("Croque Monsieur",20,"French|Sandwich|Comfort|Quick","Toasted ham and cheese sandwich finished with a creamy bechamel."),
 ("Coq au Vin",100,"French|Comfort|Gluten-Free","Chicken braised in red wine with mushrooms, bacon and pearl onions."),
 ("Creme Brulee",60,"French|Dessert|Vegetarian|Gluten-Free","Silky vanilla custard with a thin crackling caramelized sugar top."),
 ("Crepes with Berries",30,"French|Breakfast|Dessert|Vegetarian|Quick","Thin pancakes folded around fresh berries and whipped cream."),
 ("Quinoa Veggie Bowl",30,"Healthy|Vegan|Rice|Quick|Gluten-Free","Quinoa with roasted vegetables, chickpeas and a lemony tahini drizzle."),
 ("Avocado Toast with Egg",10,"Breakfast|Vegetarian|Healthy|Quick|Sandwich","Toasted sourdough topped with smashed avocado, a fried egg and chili flakes."),
 ("Overnight Oats",10,"Breakfast|Vegetarian|Healthy|Quick","Oats soaked in milk and yogurt with fruit and nuts, ready by morning."),
 ("Lentil Vegetable Soup",45,"Soup|Vegan|Healthy|Comfort|Gluten-Free","Hearty lentils simmered with carrots, celery and tomatoes."),
 ("Chicken Caesar Salad",20,"Salad|Quick|American","Romaine with grilled chicken, parmesan, croutons and creamy dressing."),
 ("Grilled Cheese Sandwich",10,"Sandwich|Vegetarian|Comfort|Quick|American","Buttery toasted bread with gooey melted cheese."),
 ("Sweet Potato Black Bean Chili",50,"Vegan|Spicy|Comfort|Healthy|Gluten-Free","Smoky chili with sweet potato, black beans and tomatoes."),
 ("Banana Bread",70,"Dessert|Baking|Vegetarian|Breakfast|Comfort","Moist loaf made with ripe bananas and a hint of cinnamon."),
 ("Chocolate Lava Cake",25,"Dessert|Baking|Vegetarian|Quick","Individual chocolate cakes with a warm molten center."),
 ("Lemon Garlic Shrimp",15,"Seafood|Quick|Gluten-Free|Healthy","Shrimp sauteed in garlic butter with lemon juice and parsley."),
 ("Grilled Fish with Salsa Verde",30,"Seafood|Grill|Healthy|Mediterranean|Gluten-Free","Whole fish grilled and served with a punchy herb and caper sauce."),
 ("Pasta Primavera",30,"Italian|Pasta|Vegetarian|Healthy|Quick","Spring vegetables tossed with pasta, olive oil and parmesan."),
 ("Beef Stew",150,"Comfort|American|Gluten-Free","Chunks of beef braised slowly with potatoes, carrots and red wine."),
 ("Vegan Buddha Bowl",25,"Healthy|Vegan|Quick|Gluten-Free","Greens, roasted sweet potato, edamame and avocado with peanut dressing."),
 ("Stuffed Bell Peppers",55,"Comfort|Mediterranean|Baking|Gluten-Free","Peppers stuffed with rice, ground meat and tomato, then baked."),
]

ARCHETYPES = {
    "spice_lover":    {"Spicy", "Indian", "Thai", "Mexican"},
    "comfort_food":   {"Comfort", "American", "Baking", "Italian"},
    "healthy_eater":  {"Healthy", "Salad", "Vegan", "Mediterranean"},
    "sweet_tooth":    {"Dessert", "Baking", "Breakfast"},
    "vegetarian":     {"Vegetarian", "Vegan"},
    "seafood_lover":  {"Seafood", "Japanese", "Mediterranean"},
    "quick_cook":     {"Quick", "Sandwich", "Salad"},
    "asian_fan":      {"Chinese", "Japanese", "Thai", "Noodles"},
}


def main(seed: int = 42, n_users: int = 150) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with open(os.path.join(HERE, "recipes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recipe_id", "title", "minutes", "tags", "description"])
        for i, (t, y, g, d) in enumerate(RECIPES, start=1):
            w.writerow([i, t, y, g, d])

    quality = np_rng.normal(0, 0.35, len(RECIPES))          # latent "general appeal"
    names = list(ARCHETYPES)
    rows = []
    for u in range(1, n_users + 1):
        archs = rng.sample(names, k=rng.choice([1, 2]))
        liked = set().union(*(ARCHETYPES[a] for a in archs))
        bias = np_rng.normal(0, 0.4)
        n_rated = rng.randint(14, 32)
        # users tend to watch what they expect to like -> sampling bias towards liked tags
        weights = [1.0 + 2.5 * len(liked & set(g.split("|"))) for (_, _, g, _) in RECIPES]
        idx = list(range(len(RECIPES)))
        chosen = set()
        while len(chosen) < n_rated:
            chosen.add(rng.choices(idx, weights=weights)[0])
        for i in sorted(chosen):
            gset = set(RECIPES[i][2].split("|"))
            frac = len(liked & gset) / len(gset)
            raw = 2.6 + 2.0 * frac + bias + quality[i] + np_rng.normal(0, 0.55)
            rows.append((u, i + 1, int(min(5, max(1, round(raw))))))

    with open(os.path.join(HERE, "ratings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "recipe_id", "rating"])
        w.writerows(rows)
    print(f"Wrote {len(RECIPES)} recipes and {len(rows)} ratings from {n_users} users.")


if __name__ == "__main__":
    main()
