from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel


class SavingsPeriod(str, Enum):
    WEEK = "week"
    MONTH = "month"


class SavingsVerdict(str, Enum):
    SAVING = "saving"
    ON_BUDGET = "on_budget"
    OVER_BUDGET = "over_budget"


class DailySavings(BaseModel):
    date: date
    planned_meals: int
    estimated_cost: Decimal
    budget: Decimal
    savings: Decimal


class SavingsProjection(BaseModel):
    # Always an estimate: average saving per planned meal, extended to
    # every meal slot in the period.
    days: int
    meals: int
    projected_savings: Decimal


class SavingsSummary(BaseModel):
    period: SavingsPeriod
    start_date: date
    end_date: date
    budget_per_meal: Decimal
    planned_meals: int
    estimated_cost_total: Decimal
    budget_total: Decimal
    total_savings: Decimal
    savings_percent: float
    verdict: SavingsVerdict
    daily: list[DailySavings]
    # None when there are too few planned meals to estimate from.
    projection: SavingsProjection | None