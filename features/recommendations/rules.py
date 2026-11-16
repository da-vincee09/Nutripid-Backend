import time
from decimal import Decimal

from sqlalchemy.orm import joinedload

from features.recommendations.utils import normalize_ingredient
from features.recommendations.models import IngredientSubstitution
from features.ingredients.repository import convert_quantity

def is_ingredient_available(
    ingredient: str,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> bool:
    # Clean the name (spacing, capitalization) so it matches the pantry keys
    normalized = normalize_ingredient(ingredient)

    # Available if the pantry has this ingredient under any unit
    return normalized in available_ingredients


def get_pantry_quantity(
    db,
    ingredient: str,
    unit: str,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> Decimal | None:
    """Returns the total pantry quantity for this ingredient, converted
    into the given unit if necessary. Returns None only if the
    ingredient isn't in the pantry at all, or is present only under
    unit(s) with no known conversion path to the requested unit
    (e.g. pantry has "pcs" but recipe needs "g").
    """
    normalized_name = normalize_ingredient(ingredient)
    normalized_unit = unit.strip().lower()

    # All units the pantry holds for this ingredient, e.g. {"g": 500, "kg": 1}
    pantry_units = available_ingredients.get(normalized_name)
    if pantry_units is None:
        return None # ingredient is not in the pantry

    # Exact unit match — no conversion needed.
    if normalized_unit in pantry_units:
        return pantry_units[normalized_unit]

    # Try converting from any unit the pantry actually has for this
    # ingredient. Sum all convertible quantities together (e.g. pantry
    # has both "500 g" AND "1 kg" of rice under separate entries —
    # both should count toward a "700 g" requirement).
    total_converted = Decimal("0")
    found_any_conversion = False

    for pantry_unit, pantry_quantity in pantry_units.items():
        # Convert this pantry entry into the recipe's unit (None if impossible)
        converted = convert_quantity(db, pantry_quantity, pantry_unit, normalized_unit)
        if converted is not None:
            total_converted += converted
            found_any_conversion = True

    if found_any_conversion:
        return total_converted

    # Ingredient is present, but no unit could be converted
    return None


def classify_ingredients(
    ingredients,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> dict[str, list[str]]:
    # Split a recipe's ingredients into "available" and "unavailable" lists
    available = []
    unavailable = []

    for ingredient in ingredients:
        name = ingredient.ingredient

        if is_ingredient_available(
            name,
            available_ingredients,
        ):
            available.append(name)
        else:
            unavailable.append(name)

    return {
        "available": available,
        "unavailable": unavailable,
    }


# ---------------------------------------------------------------------
# Substitution lookup cache
#
# get_substitute() used to run 2-3 database queries on every call, and
# it is called for every missing ingredient of every meal. Instead, the
# whole substitution table is loaded once (in a single query with joins)
# and answered from memory. The cache refreshes every 5 minutes.
# ---------------------------------------------------------------------
_substitute_cache: dict[str, str] | None = None
_substitute_cache_loaded_at: float = 0.0
_SUBSTITUTE_CACHE_TTL = 300  # seconds


def _get_substitute_map(db) -> dict[str, str]:
    """normalized ingredient name -> substitute name, loaded in one query."""
    global _substitute_cache, _substitute_cache_loaded_at

    now = time.monotonic()
    if (
        _substitute_cache is None
        or now - _substitute_cache_loaded_at > _SUBSTITUTE_CACHE_TTL
    ):
        rows = (
            db.query(IngredientSubstitution)
            .options(
                joinedload(IngredientSubstitution.ingredient_ref),
                joinedload(IngredientSubstitution.substitute_ref),
            )
            .all()
        )

        mapping: dict[str, str] = {}
        for row in rows:
            key = normalize_ingredient(row.ingredient)
            # First match wins, same as the old .first() behavior
            mapping.setdefault(key, row.substitute)

        _substitute_cache = mapping
        _substitute_cache_loaded_at = now

    return _substitute_cache


def get_substitute(
    db,
    ingredient: str,
) -> str | None:
    normalized = normalize_ingredient(ingredient)

    # Returns None if the ingredient isn't in the database or has no
    # substitute defined, same as before.
    return _get_substitute_map(db).get(normalized)


def get_effective_available_ingredients(
    db,
    ingredients,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> set[str]:
    # Name-only set for TF-IDF coverage — quantity sufficiency is a
    # separate, per-ingredient concern handled in adapt_ingredient below,
    # not part of coverage scoring.

    # Start with everything the user has in the pantry
    effective = set(available_ingredients)

    for ingredient in ingredients:
        normalized = normalize_ingredient(
            ingredient.ingredient
        )

        # Already in the pantry, nothing to do
        if normalized in effective:
            continue

        substitute = get_substitute(
            db,
            normalized,
        )

        if substitute is None:
            continue # missing and no substitute

        normalized_substitute = normalize_ingredient(
            substitute
        )

        # If the user has the substitute, count the original ingredient as covered
        if normalized_substitute in available_ingredients:
            effective.add(normalized)

    return effective


def adapt_ingredient(
    db,
    ingredient,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> dict:
    # Decide what to do with ONE recipe ingredient:
    # retain, insufficient, substitute, omit, or unavailable
    name = ingredient.ingredient
    normalized = normalize_ingredient(name)
    required_unit = ingredient.unit.strip().lower()

    # Case 1: the user has this ingredient
    if is_ingredient_available(
        normalized,
        available_ingredients,
    ):
        # How much the user has, in the recipe's unit
        pantry_quantity = get_pantry_quantity(
            db,
            normalized,
            required_unit,
            available_ingredients,
        )

        # Quantity could not be compared (no unit conversion): keep it as available
        if pantry_quantity is None:
            return {
                "ingredient": name,
                "action": "retain",
                "replacement": None,
            }

        # Enough quantity: keep it
        if pantry_quantity >= ingredient.quantity:
            return {
                "ingredient": name,
                "action": "retain",
                "replacement": None,
            }

        # Has some but not enough: report how much is missing
        return {
            "ingredient": name,
            "action": "insufficient",
            "replacement": None,
            "available_quantity": pantry_quantity,
            "required_quantity": ingredient.quantity,
            "unit": required_unit,
        }

    # Case 2: the user does not have it, so look for a substitute
    substitute = get_substitute(
        db,
        normalized,
    )

    if substitute is not None:
        normalized_substitute = normalize_ingredient(
            substitute
        )

        # Substitute exists and the user has it: replace the ingredient
        if is_ingredient_available(
            normalized_substitute,
            available_ingredients,
        ):
            return {
                "ingredient": name,
                "action": "substitute",
                "replacement": substitute,
            }

    # Case 3: no substitute, but the ingredient is optional: leave it out
    if ingredient.is_optional:
        return {
            "ingredient": name,
            "action": "omit",
            "replacement": None,
        }

    # Case 4: required, missing, no usable substitute
    return {
        "ingredient": name,
        "action": "unavailable",
        "replacement": None,
    }


def adapt_meal(
    db,
    ingredients,
    available_ingredients: dict[str, dict[str, Decimal]],
) -> dict:
    """
    Determines whether a meal can be prepared using the ingredients
    available in the user's pantry.

    It checks every ingredient in the recipe by calling
    adapt_ingredient(), then decides whether the meal is classified
    as 'adapt' or 'fallback'.

    Returns:
        A dictionary containing:
        - decision: "adapt" or "fallback"
        - ingredients: List of individual ingredient adaptation results
    """

    # Create an empty list to store the adaptation result
    # of every ingredient in the recipe.
    #
    # Each result will contain information such as:
    # ingredient name, action, replacement, and quantity details
    # when applicable.
    adaptations = []

    # Loop through every ingredient required by the meal.
    # The variable 'ingredient' represents the current ingredient
    # being checked during each iteration.
    for ingredient in ingredients:

        # Call adapt_ingredient() to determine what should happen
        # to the current ingredient based on the user's pantry.
        #
        # Possible actions returned:
        # - "retain": Ingredient is available and sufficient.
        # - "insufficient": Ingredient is available but quantity is lacking.
        # - "substitute": An available substitute can replace it.
        # - "omit": Ingredient is optional and can be left out.
        # - "unavailable": Required ingredient cannot be provided.
        #
        # The returned dictionary is appended to the adaptations list.
        adaptations.append(
            adapt_ingredient(
                db,                     # Database session used to find substitutes and convert units.
                ingredient,             # Current recipe ingredient being evaluated.
                available_ingredients,  # Ingredients and quantities in the user's pantry.
            )
        )

    # Check whether at least one ingredient has the action "unavailable".
    #
    # The 'any()' function returns True if at least one item in the
    # adaptations list satisfies the condition.
    #
    # item["action"] retrieves the action assigned to each ingredient.
    # == "unavailable" checks whether that ingredient is unavailable.
    #
    # The for loop inside any() examines each adaptation result.
    has_unavailable = any(
        item["action"] == "unavailable"
        for item in adaptations
    )

    # Determine the overall meal decision based on the ingredient results.
    if has_unavailable:

        # At least one required ingredient is missing and has no usable
        # substitute. Therefore, the meal cannot be prepared according
        # to the system's adaptation rules.
        #
        # Classify the meal as a fallback recommendation.
        decision = "fallback" 
    else:
        # No ingredient has been classified as "unavailable".
        # All ingredients have an action of retain, substitute, or omit,
        # or possibly insufficient.
        #
        # Classify the meal as adaptable according to the current
        # decision rule.
        decision = "adapt" 

    # Return the final meal decision together with the detailed
    # adaptation result for every ingredient.
    #
    # "decision" tells whether the meal is adapt or fallback.
    # "ingredients" contains the individual ingredient actions.
    return {
        "decision": decision,
        "ingredients": adaptations,
    }