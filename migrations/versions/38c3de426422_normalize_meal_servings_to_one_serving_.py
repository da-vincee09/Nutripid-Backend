"""normalize meal servings to one serving baseline

Revision ID: p8c3de426422
Revises: n2b3c4d5e6f7
Create Date: 2026-09-27 20:26:15.388806

Divides meal_ingredients.quantity, meals.estimated_cost, and meals.calories
by each meal's recorded `servings`, then sets `servings` to 1, so every
seeded meal's figures represent a single serving.

Discrete-unit ingredients (pcs/cloves) that didn't divide into a clean
kitchen fraction were resolved by rounding to the nearest common cooking
fraction (1/4, 1/3, 1/2, 2/3, 3/4) rather than left as raw decimals — see
MANUAL_OVERRIDES below. This mirrors the other documented simplifying
assumptions in the Methodology (PAL scaling, the Go-share midpoint, etc.):
a stated, defensible reconciliation rather than an unexplained rounding.

Pre-normalization values are copied into a temporary backup table before
the update runs, so downgrade() can restore them exactly.
"""
from decimal import Decimal
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'p8c3de426422'
down_revision: Union[str, Sequence[str], None] = 'n2b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Resolved by rounding each flagged row's plain quotient to the nearest of
# {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1} — see servings_normalization_review.csv
# for the raw divided values these replace. Keyed by (meal name, ingredient name).
MANUAL_OVERRIDES: dict[tuple[str, str], Decimal] = {
    ("Beef Caldereta", "bell_pepper"): Decimal("0.25"),
    ("Beef Caldereta", "carrot"): Decimal("0.33"),
    ("Beef Caldereta", "garlic"): Decimal("0.75"),
    ("Beef Caldereta", "onion"): Decimal("0.25"),
    ("Beef Caldereta", "potato"): Decimal("0.33"),
    ("Pancit Bihon", "carrot"): Decimal("0.25"),
    ("Pancit Bihon", "garlic"): Decimal("0.67"),
    ("Pancit Bihon", "onion"): Decimal("0.25"),
    ("Pork Menudo", "carrot"): Decimal("0.25"),
    ("Pork Menudo", "garlic"): Decimal("0.67"),
    ("Pork Menudo", "onion"): Decimal("0.25"),
    ("Pork Menudo", "potato"): Decimal("0.33"),
    ("Vegetable Lumpia", "onion"): Decimal("0.25"),
}


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()

    # 1. Back up current values so downgrade() can restore them exactly.
    conn.execute(sa.text("""
        CREATE TABLE _servings_normalization_backup AS
        SELECT id AS meal_id, servings, estimated_cost, calories
        FROM meals
        WHERE servings IS NOT NULL AND servings <> 1
    """))
    conn.execute(sa.text("""
        CREATE TABLE _servings_normalization_ingredient_backup AS
        SELECT mi.id AS meal_ingredient_id, mi.quantity
        FROM meal_ingredients mi
        JOIN meals m ON m.id = mi.meal_id
        WHERE m.servings IS NOT NULL AND m.servings <> 1
    """))

    # 2. Divide ingredient quantities by each meal's servings.
    conn.execute(sa.text("""
        UPDATE meal_ingredients mi
        SET quantity = ROUND(mi.quantity / m.servings, 3)
        FROM meals m
        WHERE mi.meal_id = m.id
          AND m.servings IS NOT NULL
          AND m.servings <> 1
    """))

    # 3. Apply the resolved discrete-unit overrides on top of the plain
    #    division (see MANUAL_OVERRIDES docstring above).
    for (meal_name, ingredient_name), override_qty in MANUAL_OVERRIDES.items():
        conn.execute(
            sa.text("""
                UPDATE meal_ingredients mi
                SET quantity = :q
                FROM meals m, ingredients i
                WHERE mi.meal_id = m.id
                  AND mi.ingredient_id = i.id
                  AND m.name = :meal_name
                  AND i.name = :ingredient_name
            """),
            {"q": override_qty, "meal_name": meal_name, "ingredient_name": ingredient_name},
        )

    # 4. Divide meal-level cost and calories, then set servings to 1.
    conn.execute(sa.text("""
        UPDATE meals
        SET estimated_cost = ROUND(estimated_cost / servings, 2),
            calories = ROUND(calories / servings),
            servings = 1
        WHERE servings IS NOT NULL AND servings <> 1
    """))

    # Backup tables are intentionally left in place; downgrade() drops them
    # after restoring, so they double as an audit trail until then.


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()

    conn.execute(sa.text("""
        UPDATE meal_ingredients mi
        SET quantity = b.quantity
        FROM _servings_normalization_ingredient_backup b
        WHERE mi.id = b.meal_ingredient_id
    """))
    conn.execute(sa.text("""
        UPDATE meals m
        SET servings = b.servings,
            estimated_cost = b.estimated_cost,
            calories = b.calories
        FROM _servings_normalization_backup b
        WHERE m.id = b.meal_id
    """))

    conn.execute(sa.text("DROP TABLE IF EXISTS _servings_normalization_ingredient_backup"))
    conn.execute(sa.text("DROP TABLE IF EXISTS _servings_normalization_backup"))