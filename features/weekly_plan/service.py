from datetime import timedelta
from decimal import Decimal
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from core.utils import get_app_now, is_planned_slot_in_past
from features.meal_planner import service as planner_service
from features.meal_planner.schemas import MealPlanMealSummary
from features.meals import service as meal_service
from features.nutrition.constants import is_staple_meal
from features.profiles.models.profile import Profile
from features.recommendations.service import calculate_meal_coverage
from features.weekly_plan.planner import (
    build_alternatives,
    build_weekly_plan,
)
from features.weekly_plan.schemas import (
    AlternativeMealResponse,
    PlannedSlotResponse,
    WeeklyPlanAcceptRequest,
    WeeklyPlanAcceptResponse,
    WeeklyPlanDraftResponse,
    WeeklyPlanGenerateRequest,
)


def generate_weekly_plan(
    db: Session,
    profile_id: UUID,
    request: WeeklyPlanGenerateRequest,
) -> WeeklyPlanDraftResponse:
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )

    # Staple meals (e.g. Garlic Fried Rice) are not ulam, so they are
    # not planned as a meal on their own.
    candidates = [
        c
        for c in calculate_meal_coverage(db, profile_id)
        if not is_staple_meal(c["meal"].name)
    ]

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No meals match your budget and food restrictions.",
        )

    # "Today" in the app's timezone, not the server's clock.
    start_date = request.start_date or get_app_now().date()
    end_date = start_date + timedelta(days=request.days - 1)

    plan = build_weekly_plan(
        candidates, start_date, request.days, request.seed
    )

    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No upcoming meal slots left in this date range.",
        )

    slots = [
        PlannedSlotResponse(
            planned_date=item["planned_date"],
            meal_slot=item["meal_slot"],
            meal=MealPlanMealSummary.model_validate(
                item["candidate"]["meal"]
            ),
            hybrid_score=item["candidate"]["hybrid_score"],
            coverage=float(item["candidate"]["coverage"]),
            pantry_decision=item["candidate"]["adaptation"]["decision"],
        )
        for item in plan
    ]

    alternatives = [
        AlternativeMealResponse(
            meal=MealPlanMealSummary.model_validate(c["meal"]),
            hybrid_score=c["hybrid_score"],
            pantry_decision=c["adaptation"]["decision"],
        )
        for c in build_alternatives(candidates)
    ]

    budget_per_meal = Decimal(str(profile.budget_per_meal))
    cost_total = sum(
        (Decimal(str(s.meal.estimated_cost)) for s in slots),
        Decimal("0"),
    )
    budget_total = budget_per_meal * len(slots)

    return WeeklyPlanDraftResponse(
        start_date=start_date,
        end_date=end_date,
        budget_per_meal=budget_per_meal,
        slots=slots,
        alternatives=alternatives,
        estimated_cost_total=cost_total,
        budget_total=budget_total,
        estimated_savings=budget_total - cost_total,
    )


def accept_weekly_plan(
    db: Session,
    profile_id: UUID,
    request: WeeklyPlanAcceptRequest,
) -> WeeklyPlanAcceptResponse:
    # ---- Validate everything BEFORE writing anything ----
    seen: set[tuple] = set()
    for entry in request.entries:
        key = (entry.planned_date, entry.meal_slot)
        if key in seen:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The plan has two meals in the same slot.",
            )
        seen.add(key)

        if is_planned_slot_in_past(entry.planned_date, entry.meal_slot):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The plan contains a slot that has already passed.",
            )

        if meal_service.get_meal_by_id(db, entry.meal_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="A meal in the plan was not found.",
            )

    # ---- Existing entries in the same date range ----
    dates = [e.planned_date for e in request.entries]
    existing = planner_service.get_meal_plan_entries(
        db, profile_id, min(dates), max(dates)
    )
    occupied: dict[tuple, list] = {}
    for item in existing:
        occupied.setdefault(
            (item.planned_date, item.meal_slot), []
        ).append(item)

    saved = 0
    skipped = 0

    for entry in request.entries:
        key = (entry.planned_date, entry.meal_slot)
        taken = occupied.get(key, [])

        if taken and not request.replace_existing:
            skipped += 1  # never overwrite the user's own entries by default
            continue

        for old in taken:
            planner_service.delete_meal_plan_entry(db, old)

        planner_service.create_meal_plan_entry(db, profile_id, entry)
        saved += 1

    return WeeklyPlanAcceptResponse(
        saved_count=saved,
        skipped_count=skipped,
    )