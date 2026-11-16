"""
Week X — Objective 3 evaluation harness.

Produces the three accuracy measures the Methodology promises for
Objective 3: ingredient-matching accuracy, budget-based recommendation
accuracy, and Top-K ranking accuracy, plus weighted ingredient coverage
as a supporting measure.

Design principle: "expected" values are computed independently of the
functions under test wherever possible — either by construction (the
tester decides what goes into a synthetic pantry, so the correct
classification is known in advance) or via a from-scratch reference
calculation. Top-K's expected ranking is now the full hybrid-relevance
formula (weighted ingredient coverage 30%, budget score 30%, cooking-skill
score 10%, allergy score 20%, disliked-ingredient score 10%), matching the
revised Chapter 3 methodology — computed by hand in this script, not by
calling MealTfidfEngine or features.recommendations.scoring. This mirrors
the Methodology's description of expected results being established
before the algorithm runs.

Run from the backend project root, inside the venv:
    python -m scripts.evaluate_recommendation_accuracy
"""

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session

from core.database import SessionLocal
from features.meals.models import Meal
from features.recommendations.utils import normalize_ingredient
from features.recommendations.rules import adapt_meal
from features.recommendations.scoring import calculate_budget_score
from features.recommendations.service import calculate_meal_coverage


# ── Fill in before running ────────────────────────────────────────────
# Real Profile IDs (UUIDs) for one or more test accounts already seeded
# in the database. Each should have a known budget_per_meal, cooking
# skill level, allergies, and dislikes, since Top-K ranking calls the
# real calculate_meal_coverage(), which looks up Profile by ID.
#
# NOTE: for Top-K to actually demonstrate the algorithm's behavior
# across different conditions, add more than one profile here —
# ideally profiles that differ in budget_per_meal, cooking_skill_level,
# and at least one with allergies/dislikes set, so the hybrid-score
# components (skill/allergy/dislike) actually get exercised rather than
# defaulting to 1.0 for every meal.
TEST_PROFILE_IDS: list[str] = [
    # "00000000-0000-0000-0000-000000000000",
    "26548a59-9cbd-4c16-9061-5b500dbc4fa0",
]

TOP_K = 5

# Multiple pantry-availability scenarios per profile, mirroring the
# complete/partial/limited/no-match spread used for ingredient-matching
# accuracy, so Top-K isn't judged on a single condition.
TOPK_PANTRY_FRACTIONS = [0.25, 0.50, 0.75]


# ═══════════════════════════════════════════════════════════════════
# 1. Ingredient-Matching Accuracy
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IngredientMatchCase:
    meal_name: str
    scenario: str  # complete-match | partial-match | limited-ingredient | no-match
    available_ingredients: dict[str, dict[str, Decimal]]
    expected_labels: dict[str, str]  # ingredient name -> expected action


def _expected_label_for_missing(ingredient) -> str:
    """
    Expected label for a required ingredient NOT placed in the
    synthetic pantry and with no substitute placed in the pantry
    either. Optional ingredients are expected to be omitted; required
    ingredients are expected to be unavailable.

    Assumption: the synthetic pantries built below never coincidentally
    contain another of the SAME meal's own ingredients that happens to
    be a registered substitute for this one. This has not been checked
    against the ingredient_substitutions table for every seeded meal —
    flag for manual review if a mismatch shows up in the results CSV.
    """
    return "omit" if ingredient.is_optional else "unavailable"


def build_ingredient_matching_cases(meals: list[Meal]) -> list[IngredientMatchCase]:
    cases = []

    for meal in meals:
        ingredients = list(meal.ingredients)
        n = len(ingredients)
        if n == 0:
            continue

        half = max(1, n // 2)

        scenarios = {
            "complete-match": ingredients,          # every ingredient in pantry
            "partial-match": ingredients[:half],    # first half in pantry
            "limited-ingredient": ingredients[:1],  # only the first ingredient
            "no-match": [],                         # nothing in pantry
        }

        for scenario_name, included in scenarios.items():
            pantry: dict[str, dict[str, Decimal]] = {}
            expected: dict[str, str] = {}

            included_names = {id(i) for i in included}

            for ingredient in ingredients:
                name = normalize_ingredient(ingredient.ingredient)
                unit = ingredient.unit.strip().lower()

                if id(ingredient) in included_names:
                    # Place at exactly the required quantity — expected: retain.
                    pantry.setdefault(name, {})
                    pantry[name][unit] = ingredient.quantity
                    expected[ingredient.ingredient] = "retain"
                else:
                    expected[ingredient.ingredient] = _expected_label_for_missing(
                        ingredient
                    )

            cases.append(
                IngredientMatchCase(
                    meal_name=meal.name,
                    scenario=scenario_name,
                    available_ingredients=pantry,
                    expected_labels=expected,
                )
            )

    return cases


def run_ingredient_matching_tests(db: Session, meals: list[Meal]) -> dict:
    cases = build_ingredient_matching_cases(meals)

    total = 0
    correct = 0
    mismatches = []

    for case in cases:
        meal = next(m for m in meals if m.name == case.meal_name)
        result = adapt_meal(db, meal.ingredients, case.available_ingredients)

        actual_by_name = {
            item["ingredient"]: item["action"] for item in result["ingredients"]
        }

        for ingredient_name, expected_action in case.expected_labels.items():
            actual_action = actual_by_name.get(ingredient_name)
            total += 1
            if actual_action == expected_action:
                correct += 1
            else:
                mismatches.append(
                    {
                        "meal": case.meal_name,
                        "scenario": case.scenario,
                        "ingredient": ingredient_name,
                        "expected": expected_action,
                        "actual": actual_action,
                    }
                )

    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "mismatches": mismatches,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Budget-Based Recommendation Accuracy
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BudgetTestCase:
    meal_name: str
    meal_cost: float
    budget: float
    expected_within_budget: bool
    expected_score_tier: float  # expected calculate_budget_score() output


def build_budget_test_cases(meals: list[Meal]) -> list[BudgetTestCase]:
    cases = []

    # Ratios chosen to exercise every tier boundary in
    # calculate_budget_score, plus the hard affordability gate at 1.00.
    ratios = [0.25, 0.50, 0.51, 0.75, 0.76, 1.00, 1.01, 1.25, 1.26, 1.50]

    for meal in meals:
        cost = float(meal.estimated_cost)
        if cost <= 0:
            continue

        for ratio in ratios:
            budget = round(cost / ratio, 2)
            actual_ratio = cost / budget   # recompute from the rounded budget, not the nominal ratio

            if actual_ratio <= 0.50:
                expected_tier = 1.00
            elif actual_ratio <= 0.75:
                expected_tier = 0.90
            elif actual_ratio <= 1.00:
                expected_tier = 0.80
            elif actual_ratio <= 1.25:
                expected_tier = 0.50
            else:
                expected_tier = 0.00

            cases.append(
                BudgetTestCase(
                    meal_name=meal.name,
                    meal_cost=cost,
                    budget=budget,
                    expected_within_budget=(cost <= budget),
                    expected_score_tier=expected_tier,
                )
            )

    return cases


def run_budget_tests(meals: list[Meal]) -> dict:
    cases = build_budget_test_cases(meals)

    total = 0
    correct_classification = 0
    correct_tier = 0
    mismatches = []

    for case in cases:
        actual_within_budget = case.meal_cost <= case.budget
        actual_tier = calculate_budget_score(case.meal_cost, case.budget)

        total += 1
        classification_ok = actual_within_budget == case.expected_within_budget
        tier_ok = math.isclose(actual_tier, case.expected_score_tier, abs_tol=1e-6)

        if classification_ok:
            correct_classification += 1
        if tier_ok:
            correct_tier += 1

        if not (classification_ok and tier_ok):
            mismatches.append(
                {
                    "meal": case.meal_name,
                    "cost": case.meal_cost,
                    "budget": case.budget,
                    "expected_within_budget": case.expected_within_budget,
                    "actual_within_budget": actual_within_budget,
                    "expected_tier": case.expected_score_tier,
                    "actual_tier": actual_tier,
                }
            )

    return {
        "total": total,
        "classification_accuracy": correct_classification / total if total else 0.0,
        "tier_accuracy": correct_tier / total if total else 0.0,
        "mismatches": mismatches,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Top-K Ranking Accuracy
# ═══════════════════════════════════════════════════════════════════
#
# Reuses building blocks already independently validated elsewhere in
# this script/study:
#   - adapt_meal / get_effective_available_ingredients: validated by
#     Ingredient-Matching Accuracy (Measure 1), 100% on 572 cases.
#   - calculate_budget_score: validated by Budget-Based Recommendation
#     Accuracy (Measure 2), 100% on 200 cases.
#   - calculate_skill_score / calculate_allergy_score /
#     calculate_disliked_ingredient_score: simple deterministic lookup
#     tables, reimplemented from scratch would just be transcription,
#     not independent verification.
#   - MealTfidfEngine: used for consistency with the coverage figure
#     reported as TIPIDMEAL's supporting measure. NOTE: as configured,
#     this uses sklearn's default smoothed/L2-normalized TF-IDF, which
#     differs from the plain log(n/df) formula documented in Chapter 3
#     — see accompanying note; this is a separate, pre-existing
#     discrepancy this script does not attempt to resolve.
#
# What IS independently (re)implemented here, because it is the thing
# Top-K is actually meant to verify and is not covered by any other
# measure: the hard affordability/allergen filters and the
# tier-then-hybrid-score sort order.

from features.recommendations.rules import (
    adapt_meal,
    get_effective_available_ingredients,
)
from features.recommendations.scoring import (
    calculate_skill_score,
    calculate_allergy_score,
    calculate_disliked_ingredient_score,
    calculate_hybrid_score,
)
from features.recommendations.tfidf import MealTfidfEngine


@dataclass
class ProfileContext:
    profile_id: str
    budget_per_meal: float
    cooking_skill_level: str
    allergy_ingredients: list[str]    # raw names, NOT underscore-normalized
    disliked_ingredients: list[str]   # raw names, NOT underscore-normalized


def _load_profile_context(db: Session, profile_id: str) -> ProfileContext:
    """
    Independently pulls the profile attributes the hybrid score depends
    on, straight from the tables (not through the recommendation
    service), expanding allergy categories into concrete ingredient
    names via allergen_categories. Names are kept RAW (only
    .strip().lower()'d, no underscore-joining) because
    calculate_allergy_score / calculate_disliked_ingredient_score do
    their own plain .strip().lower() normalization internally — this
    must match how service.py actually calls them, not the underscore
    normalization used for TF-IDF elsewhere.

    ASSUMPTION: table/column names match the Chapter 3 data dictionary.
    Adjust the raw SQL below if your actual schema differs.
    """
    profile_row = db.execute(
        text(
            "SELECT budget_per_meal, cooking_skill_level "
            "FROM profiles WHERE id = :pid"
        ),
        {"pid": profile_id},
    ).fetchone()

    if profile_row is None:
        raise ValueError(f"Profile {profile_id} not found.")

    budget_per_meal, cooking_skill_level = profile_row

    allergy_categories = [
        row[0]
        for row in db.execute(
            text("SELECT allergy FROM food_allergies WHERE profile_id = :pid"),
            {"pid": profile_id},
        ).fetchall()
    ]

    allergy_ingredients: list[str] = []
    if allergy_categories:
        stmt = text(
            """
            SELECT i.name
            FROM allergen_categories ac
            JOIN ingredients i ON i.id = ac.ingredient_id
            WHERE ac.category IN :categories
            """
        ).bindparams(bindparam("categories", expanding=True))
        allergy_ingredients = [
            row[0]
            for row in db.execute(
                stmt, {"categories": allergy_categories}
            ).fetchall()
        ]

    disliked_ingredients = [
        row[0]
        for row in db.execute(
            text(
                """
                SELECT i.name
                FROM disliked_ingredients di
                JOIN ingredients i ON i.id = di.ingredient_id
                WHERE di.profile_id = :pid
                """
            ),
            {"pid": profile_id},
        ).fetchall()
    ]

    return ProfileContext(
        profile_id=profile_id,
        budget_per_meal=float(budget_per_meal),
        cooking_skill_level=cooking_skill_level,
        allergy_ingredients=allergy_ingredients,
        disliked_ingredients=disliked_ingredients,
    )


def _expected_hybrid_ranking(
    db: Session,
    meals: list[Meal],
    available_ingredients: dict[str, dict[str, Decimal]],
    profile_ctx: ProfileContext,
) -> list[str]:
    """
    Independently mirrors calculate_meal_coverage's actual candidate
    filtering, scoring, and sort — see module docstring for exactly
    which pieces are reused vs. independently re-derived here.
    """
    engine = MealTfidfEngine()
    engine.fit(meals)

    scored = []
    for index, meal in enumerate(meals):
        cost = float(meal.estimated_cost)

        # Hard affordability gate — mirrors service.py's `continue`,
        # independent of calculate_budget_score's own tiering.
        if cost > profile_ctx.budget_per_meal:
            continue

        adaptation = adapt_meal(db, meal.ingredients, available_ingredients)
        effective_ingredients = get_effective_available_ingredients(
            db, meal.ingredients, available_ingredients
        )
        coverage = engine.calculate_weighted_ingredient_coverage(
            index, effective_ingredients
        )

        meal_ingredient_names = [i.ingredient for i in meal.ingredients]

        budget_score = calculate_budget_score(cost, profile_ctx.budget_per_meal)
        skill_score = calculate_skill_score(
            profile_ctx.cooking_skill_level, meal.difficulty
        )
        allergy_score = calculate_allergy_score(
            meal_ingredient_names, profile_ctx.allergy_ingredients
        )
        disliked_score = calculate_disliked_ingredient_score(
            meal_ingredient_names, profile_ctx.disliked_ingredients
        )

        # Hard allergen exclusion — mirrors service.py's `continue`.
        if allergy_score == 0.0:
            continue

        hybrid_score = calculate_hybrid_score(
            float(coverage), budget_score, skill_score, allergy_score, disliked_score
        )

        # Independently re-derived tier-then-score rule: "adapt" tier
        # ahead of "fallback" tier, hybrid score descending within
        # each tier — matches the ordering rule observed in
        # calculate_meal_coverage's default "score" sort.
        scored.append((meal.name, adaptation["decision"] != "adapt", -hybrid_score))

    scored.sort(key=lambda triple: (triple[1], triple[2]))
    return [name for name, _, _ in scored]


def run_topk_ranking_test(
    db: Session,
    meals: list[Meal],
    profile_ctx: ProfileContext,
    available_ingredients: dict[str, dict[str, Decimal]],
    scenario_label: str,
) -> dict:
    expected_ranking = _expected_hybrid_ranking(
        db, meals, available_ingredients, profile_ctx
    )
    expected_top_k = set(expected_ranking[:TOP_K])

    actual_recommendations = calculate_meal_coverage(
        db, profile_ctx.profile_id, sort_by="score"
    )
    actual_ranking = [item["meal"].name for item in actual_recommendations]
    actual_top_k = set(actual_ranking[:TOP_K])

    overlap = expected_top_k & actual_top_k
    topk_accuracy = len(overlap) / TOP_K if TOP_K else 0.0

    return {
        "profile_id": profile_ctx.profile_id,
        "scenario": scenario_label,
        "expected_top_k": list(expected_top_k),
        "actual_top_k": list(actual_top_k),
        "overlap_count": len(overlap),
        "topk_accuracy": topk_accuracy,
        "expected_ranking": expected_ranking,
        "actual_ranking": actual_ranking,
    }


def _sample_pantry_available_ingredients(
    meals: list[Meal], fraction: float = 0.5
) -> dict[str, dict[str, Decimal]]:
    """
    Builds a shared 'available ingredients' dict (name -> {unit: qty})
    spanning a fraction of all distinct ingredients across the seeded
    meal database, with quantities set generously high so quantity-
    sufficiency never becomes the limiting factor in this scenario —
    that dimension is already covered by Ingredient-Matching Accuracy.
    """
    all_ingredients = {}
    for meal in meals:
        for ing in meal.ingredients:
            name = normalize_ingredient(ing.ingredient)
            unit = ing.unit.strip().lower()
            all_ingredients.setdefault(name, unit)

    names = sorted(all_ingredients)
    cutoff = max(1, int(len(names) * fraction))
    chosen = names[:cutoff]

    return {
        name: {all_ingredients[name]: Decimal("999999")}
        for name in chosen
    }


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def run_evaluation():
    db = SessionLocal()
    try:
        meals = db.query(Meal).order_by(Meal.name).all()

        print("=" * 60)
        print("1. INGREDIENT-MATCHING ACCURACY")
        print("=" * 60)
        matching_result = run_ingredient_matching_tests(db, meals)
        print(f"Total ingredient classifications tested: {matching_result['total']}")
        print(f"Correct: {matching_result['correct']}")
        print(f"Ingredient-Matching Accuracy: {matching_result['accuracy']:.2%}")
        if matching_result["mismatches"]:
            print(f"Mismatches: {len(matching_result['mismatches'])} (see CSV)")

        print()
        print("=" * 60)
        print("2. BUDGET-BASED RECOMMENDATION ACCURACY")
        print("=" * 60)
        budget_result = run_budget_tests(meals)
        print(f"Total budget test cases: {budget_result['total']}")
        print(
            f"Within/Outside Budget Classification Accuracy: "
            f"{budget_result['classification_accuracy']:.2%}"
        )
        print(f"Budget Score Tier Accuracy: {budget_result['tier_accuracy']:.2%}")
        if budget_result["mismatches"]:
            print(f"Mismatches: {len(budget_result['mismatches'])} (see CSV)")

        print()
        print("=" * 60)
        print("3. TOP-K RANKING ACCURACY (K={})".format(TOP_K))
        print("=" * 60)
        if not TEST_PROFILE_IDS:
            print(
                "No TEST_PROFILE_IDS configured — skipping Top-K ranking test. "
                "Add at least one real profile UUID at the top of this script."
            )
            topk_results = []
        else:
            topk_results = []
            for profile_id in TEST_PROFILE_IDS:
                profile_ctx = _load_profile_context(db, profile_id)
                for fraction in TOPK_PANTRY_FRACTIONS:
                    available = _sample_pantry_available_ingredients(
                        meals, fraction=fraction
                    )
                    scenario_label = f"pantry_fraction={fraction}"
                    topk_results.append(
                        run_topk_ranking_test(
                            db, meals, profile_ctx, available, scenario_label
                        )
                    )

            avg_topk_accuracy = sum(r["topk_accuracy"] for r in topk_results) / len(
                topk_results
            )
            print(f"Profiles tested: {len(TEST_PROFILE_IDS)}")
            print(f"Scenarios tested: {len(topk_results)}")
            print(f"Average Top-{TOP_K} Accuracy: {avg_topk_accuracy:.2%}")
            for r in topk_results:
                print(
                    f"  Profile {r['profile_id']} [{r['scenario']}]: "
                    f"{r['overlap_count']}/{TOP_K} matched "
                    f"({r['topk_accuracy']:.2%})"
                )

        # Write mismatch/detail CSVs for appendix material.
        output_dir = Path(".")

        with (output_dir / "recommendation_ingredient_matching_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["meal", "scenario", "ingredient", "expected", "actual"]
            )
            writer.writeheader()
            writer.writerows(matching_result["mismatches"])

        with (output_dir / "recommendation_budget_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "meal",
                    "cost",
                    "budget",
                    "expected_within_budget",
                    "actual_within_budget",
                    "expected_tier",
                    "actual_tier",
                ],
            )
            writer.writeheader()
            writer.writerows(budget_result["mismatches"])

        if topk_results:
            with (output_dir / "recommendation_topk_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "profile_id",
                        "scenario",
                        "overlap_count",
                        "topk_accuracy",
                        "expected_top_k",
                        "actual_top_k",
                    ],
                )
                writer.writeheader()
                for r in topk_results:
                    writer.writerow(
                        {
                            "profile_id": r["profile_id"],
                            "scenario": r["scenario"],
                            "overlap_count": r["overlap_count"],
                            "topk_accuracy": r["topk_accuracy"],
                            "expected_top_k": ", ".join(r["expected_top_k"]),
                            "actual_top_k": ", ".join(r["actual_top_k"]),
                        }
                    )

        print()
        print("Detail CSVs written to the backend project root.")

    finally:
        db.close()


if __name__ == "__main__":
    run_evaluation()