def calculate_budget_score(
    meal_cost: float,
    budget_per_meal: float,
) -> float:
    """
    Score how well a meal's cost fits the user's budget per meal (0.0 to 1.0).
    The cheaper the meal compared with the budget, the higher the score.
    """

    # No valid budget set, so the meal cannot be scored as affordable
    if budget_per_meal <= 0:
        return 0.0

    # Share of the budget the meal uses (0.5 = half, 1.0 = exactly the budget)
    ratio = meal_cost / budget_per_meal

    # Costs half the budget or less: best score
    if ratio <= 0.50:
        return 1.00

    # Up to 75% of the budget
    if ratio <= 0.75:
        return 0.90

    # Up to 100%: still within budget
    if ratio <= 1.00:
        return 0.80

    # Slightly over budget (up to 125%): heavily reduced score
    if ratio <= 1.25:
        return 0.50

    # More than 125% of the budget: no points
    return 0.00


def calculate_skill_score(
    user_skill: str,
    meal_difficulty: str,
) -> float:
    """
    Score how well the meal's difficulty matches the user's cooking skill.
    A skill mismatch lowers the score but does not zero it out, and
    an advanced user is never penalized.
    """

    # Clean the text so "Beginner " and "beginner" are treated the same
    skill = user_skill.strip().lower()
    difficulty = meal_difficulty.strip().lower()

    # Lookup table: user skill -> meal difficulty -> score
    compatibility = {
        "beginner": {
            "easy": 1.00,
            "medium": 0.60,
            "hard": 0.20,
        },
        "intermediate": {
            "easy": 1.00,
            "medium": 1.00,
            "hard": 0.60,
        },
        "advanced": {
            "easy": 1.00,
            "medium": 1.00,
            "hard": 1.00,
        },
    }

    # Look up the score; unknown skill or difficulty falls back to 0.0
    return compatibility.get(
        skill,
        {},
    ).get(
        difficulty,
        0.0,
    )


def calculate_allergy_score(
    meal_ingredients: list[str],
    allergy_ingredients: set[str],
) -> float:
    """
    allergy_ingredients is the EXPANDED set of actual ingredient names
    the profile is allergic to (already resolved from allergy category
    names like "Shellfish" via allergen_categories) — not the raw
    category strings themselves.
    """

    # Clean the meal's ingredient names (trim spaces, lowercase)
    normalized_ingredients = {
        ingredient.strip().lower()
        for ingredient in meal_ingredients
    }

    # Clean the allergen names the same way so they can be compared
    normalized_allergy_ingredients = {
        ingredient.strip().lower()
        for ingredient in allergy_ingredients
    }

    # Any overlap means the meal contains an allergen: score 0.0
    if normalized_ingredients.intersection(
        normalized_allergy_ingredients
    ):
        return 0.0

    # No allergens found: safe
    return 1.0


def calculate_disliked_ingredient_score(
    meal_ingredients: list[str],
    disliked_ingredients: list[str],
) -> float:
    """
    Score a meal by how many of the user's disliked ingredients it contains.
    More disliked ingredients lower the score, but only down to 0.70.
    """

    # Clean the meal's ingredient names
    normalized_ingredients = {
        ingredient.strip().lower()
        for ingredient in meal_ingredients
    }

    # Clean the disliked ingredient names
    normalized_disliked = {
        ingredient.strip().lower()
        for ingredient in disliked_ingredients
    }

    # How many disliked ingredients appear in this meal
    disliked_count = len(
        normalized_ingredients.intersection(
            normalized_disliked
        )
    )

    # None: full score
    if disliked_count == 0:
        return 1.00

    # One disliked ingredient
    if disliked_count == 1:
        return 0.90

    # Two disliked ingredients
    if disliked_count == 2:
        return 0.80

    # Three or more: lowest score, but not zero
    return 0.70


def calculate_hybrid_score(
    coverage: float,
    budget_score: float,
    skill_score: float,
    allergy_score: float,
    disliked_score: float,
) -> float:
    """
    Combine the five component scores into one final relevance score.
    Weights: coverage 30%, budget 30%, allergy 20%, skill 10%, dislikes 10%.
    Meals are ranked by this value.
    """

    return (
        coverage * 0.30 # TF-IDF weighted ingredient coverage
        + budget_score * 0.30 # how well the cost fits the budget
        + skill_score * 0.10 # cooking skill vs. meal difficulty
        + allergy_score * 0.20 # allergy safety
        + disliked_score * 0.10 # avoidance of disliked ingredients
    )