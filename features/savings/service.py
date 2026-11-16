import calendar
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy.orm import Session

from core.constants import MEAL_SLOT_CUTOFFS
from core.utils import get_app_now
from features.meal_planner import service as planner_service
from features.profiles.models.profile import Profile
from features.savings.schemas import (
    DailySavings,
    SavingsPeriod,
    SavingsProjection,
    SavingsSummary,
    SavingsVerdict,
)

MEALS_PER_DAY = len(MEAL_SLOT_CUTOFFS)  # breakfast, lunch, dinner = 3
ON_BUDGET_MARGIN = Decimal("0.05")  # +/-5% of the budget counts as on budget
MIN_MEALS_FOR_PROJECTION = 3

_CENT = Decimal("0.01")


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _period_range(period: SavingsPeriod, today: date) -> tuple[date, date]:
    if period == SavingsPeriod.MONTH:
        last_day = calendar.monthrange(today.year, today.month)[1]
        return today.replace(day=1), today.replace(day=last_day)

    # Week: Monday to Sunday, same as the meal planner.
    monday = today - timedelta(days=today.weekday())
    return monday, monday + timedelta(days=6)


def _verdict(total_savings: Decimal, budget_total: Decimal) -> SavingsVerdict:
    margin = budget_total * ON_BUDGET_MARGIN

    if abs(total_savings) <= margin:
        return SavingsVerdict.ON_BUDGET
    if total_savings > 0:
        return SavingsVerdict.SAVING
    return SavingsVerdict.OVER_BUDGET


def get_savings_summary(
    db: Session,
    profile: Profile,
    period: SavingsPeriod,
) -> SavingsSummary:
    # "This week/month" in the app's timezone, not the server's clock.
    today = get_app_now().date()
    start_date, end_date = _period_range(period, today)
    total_days = (end_date - start_date).days + 1

    budget_per_meal = Decimal(str(profile.budget_per_meal))

    entries = planner_service.get_meal_plan_entries(
        db, profile.id, start_date, end_date
    )

    cost_by_day: dict[date, Decimal] = defaultdict(lambda: Decimal("0"))
    count_by_day: dict[date, int] = defaultdict(int)

    for entry in entries:
        cost_by_day[entry.planned_date] += Decimal(str(entry.meal.estimated_cost))
        count_by_day[entry.planned_date] += 1

    daily: list[DailySavings] = []
    for offset in range(total_days):
        day = start_date + timedelta(days=offset)
        meals = count_by_day.get(day, 0)
        cost = cost_by_day.get(day, Decimal("0"))
        budget = budget_per_meal * meals

        daily.append(
            DailySavings(
                date=day,
                planned_meals=meals,
                estimated_cost=_money(cost),
                budget=_money(budget),
                savings=_money(budget - cost),
            )
        )

    planned_meals = len(entries)
    cost_total = sum(cost_by_day.values(), Decimal("0"))
    budget_total = budget_per_meal * planned_meals
    total_savings = budget_total - cost_total

    savings_percent = (
        float(total_savings / budget_total * 100) if budget_total > 0 else 0.0
    )

    projection = None
    if planned_meals >= MIN_MEALS_FOR_PROJECTION:
        avg_saving = total_savings / planned_meals
        projected_meals = MEALS_PER_DAY * total_days
        projection = SavingsProjection(
            days=total_days,
            meals=projected_meals,
            projected_savings=_money(avg_saving * projected_meals),
        )

    return SavingsSummary(
        period=period,
        start_date=start_date,
        end_date=end_date,
        budget_per_meal=_money(budget_per_meal),
        planned_meals=planned_meals,
        estimated_cost_total=_money(cost_total),
        budget_total=_money(budget_total),
        total_savings=_money(total_savings),
        savings_percent=round(savings_percent, 1),
        verdict=_verdict(total_savings, budget_total),
        daily=daily,
        projection=projection,
    )