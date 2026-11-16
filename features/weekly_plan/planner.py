import random
from collections import Counter
from datetime import date, timedelta

from core.constants import MEAL_SLOT_CUTOFFS
from core.utils import is_planned_slot_in_past

# Slot names and order come from the app's own constants (dict order is
# insertion order: breakfast, lunch, dinner).
MEAL_SLOTS = tuple(MEAL_SLOT_CUTOFFS)

MAX_USES_PER_WEEK = 2   # a meal appears at most twice in a plan
MIN_GAP_DAYS = 2        # same meal not again within 2 days
JITTER = 0.05           # small random nudge so "regenerate" differs
ALTERNATIVES_LIMIT = 12


def _allowed(
    candidate: dict,
    slot_date: date,
    uses: Counter,
    last_used: dict,
    level: int,
) -> bool:
    """
    Variety rules, relaxed step by step only when the pool is too small:
      0: under the weekly cap AND at least MIN_GAP_DAYS since last use
      1: under the weekly cap, just not the same day
      2: not the same day
      3: anything
    """
    meal_id = candidate["meal"].id
    used = uses[meal_id]
    last = last_used.get(meal_id)
    days_since = (slot_date - last).days if last else None

    if level == 0:
        return used < MAX_USES_PER_WEEK and (
            last is None or days_since >= MIN_GAP_DAYS
        )
    if level == 1:
        return used < MAX_USES_PER_WEEK and (last is None or days_since > 0)
    if level == 2:
        return last is None or days_since > 0
    return True


def _pick(candidates, slot_date, uses, last_used, rng):
    for level in range(4):
        eligible = [
            c for c in candidates
            if _allowed(c, slot_date, uses, last_used, level)
        ]
        if eligible:
            return max(
                eligible,
                key=lambda c: c["hybrid_score"] + rng.uniform(0, JITTER),
            )
    return None


def build_weekly_plan(
    candidates: list[dict],
    start_date: date,
    days: int = 7,
    seed: int | None = None,
) -> list[dict]:
    """
    Greedy selection: for each slot in date order, take the
    highest-scoring candidate that passes the variety rules.

    `candidates` are recommendation dicts from calculate_meal_coverage,
    which already enforces affordability and allergy exclusion.
    Slots that have already passed today are skipped.
    """
    rng = random.Random(seed)
    uses: Counter = Counter()
    last_used: dict = {}
    plan: list[dict] = []

    for offset in range(days):
        slot_date = start_date + timedelta(days=offset)

        for slot in MEAL_SLOTS:
            if is_planned_slot_in_past(slot_date, slot):
                continue

            chosen = _pick(candidates, slot_date, uses, last_used, rng)
            if chosen is None:
                continue

            meal_id = chosen["meal"].id
            uses[meal_id] += 1
            last_used[meal_id] = slot_date

            plan.append(
                {
                    "planned_date": slot_date,
                    "meal_slot": slot,
                    "candidate": chosen,
                }
            )

    return plan


def build_alternatives(candidates: list[dict]) -> list[dict]:
    ranked = sorted(
        candidates, key=lambda c: c["hybrid_score"], reverse=True
    )
    return ranked[:ALTERNATIVES_LIMIT]