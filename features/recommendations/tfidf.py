import math
from collections import Counter

from features.recommendations.utils import normalize_ingredient


class MealTfidfEngine:
    """
    From-scratch TF-IDF implementation matching the formula documented
    in Chapter 3 exactly:
        TF  = occurrences of ingredient in recipe / total ingredients in recipe
        IDF = log(total recipes / recipes containing the ingredient)
        weighted coverage = sum(TF-IDF of matched ingredients)
                             / sum(TF-IDF of all recipe ingredients)

    Replaces the previous sklearn TfidfVectorizer-based version, which
    used sklearn's smoothed/normalized defaults (idf = ln((1+n)/(1+df)) + 1,
    L2-normalized vectors) — a mismatch against the documented formula
    that wasn't a deliberate design choice, just an unexamined default.
    """

    def __init__(self):
        self.meals = []
        self.documents: dict[int, list[str]] = {}
        self.doc_freq: Counter = Counter()
        self.n_recipes = 0

    def fit(self, meals):
        self.meals = meals
        self.documents = {
            index: [
                normalize_ingredient(ingredient.ingredient)
                for ingredient in meal.ingredients
            ]
            for index, meal in enumerate(meals)
        }
        self.n_recipes = len(self.documents)

        self.doc_freq = Counter()
        for tokens in self.documents.values(): 
            for token in set(tokens): # Counts if the ingredients appear in the list
                self.doc_freq[token] += 1

        return self.documents

    # What the code does, in 4 steps
    #   1. Take one recipe and list its ingredients.
    #   2. Give each ingredient a weight. Common ingredients get a small number, and rare ones get a big number.
    #   3. Add up all the weights. Call this the total.
    #   4. Add up only the weights of ingredients the student has. Call this the matched part.
    def calculate_weighted_ingredient_coverage(
        self,
        meal_index: int,
        available_ingredients: set[str],
    ) -> float:
        """
        Return how much of a meal the user can already make, from 0.0 to 1.0.

        Each ingredient is weighted by TF-IDF, so rare, distinctive ingredients
        count more than common ones. Coverage = weight of the ingredients the
        user has / total weight of all the meal's ingredients.
        This is only one input to the hybrid score; it does not rank meals.
        """
        # Stop early if no meals were loaded into the engine
        if not self.documents:
            raise ValueError("TF-IDF engine has not been fitted.")

        # Ingredient list (tokens) of the selected meal
        tokens = self.documents[meal_index]
        total_terms = len(tokens)  # number of ingredients, used as the TF denominator
        if total_terms == 0:
            return 0.0  # meal has no ingredients, so nothing can be covered

        # Count how many times each ingredient appears in this meal
        term_counts = Counter(tokens)
        total_weight = 0.0    # weight of ALL ingredients in the meal (denominator)
        matched_weight = 0.0  # weight of ingredients the user HAS (numerator)

        for token, count in term_counts.items():
            # TF: how often the ingredient appears in this meal
            tf = count / total_terms
            # IDF: rarer ingredients across all recipes get a higher value
            idf = math.log(self.n_recipes / self.doc_freq[token])
            # TF-IDF weight of this ingredient
            weight = tf * idf
            total_weight += weight
            # Count the weight only if the user has the ingredient
            if token in available_ingredients:
                matched_weight += weight

        # Coverage ratio; return 0.0 if the total weight is zero to avoid dividing by zero
        return matched_weight / total_weight if total_weight else 0.0