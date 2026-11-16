from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from features.meal_planner.schemas import (
    MealPlanEntryCreate,
    MealPlanMealSummary,
)


class WeeklyPlanGenerateRequest(BaseModel):
    # Client sends its local date so "today" is correct for the user.
    start_date: date | None = None
    days: int = Field(default=7, ge=1, le=7)
    # Optional: makes a draft reproducible (useful for testing).
    seed: int | None = None


class PlannedSlotResponse(BaseModel):
    planned_date: date
    meal_slot: str
    meal: MealPlanMealSummary
    hybrid_score: float
    coverage: float
    pantry_decision: str  # "adapt" | "fallback"


class AlternativeMealResponse(BaseModel):
    meal: MealPlanMealSummary
    hybrid_score: float
    pantry_decision: str


class WeeklyPlanDraftResponse(BaseModel):
    start_date: date
    end_date: date
    budget_per_meal: Decimal
    slots: list[PlannedSlotResponse]
    # Ranked pool the app uses for "swap this meal".
    alternatives: list[AlternativeMealResponse]
    estimated_cost_total: Decimal
    budget_total: Decimal
    estimated_savings: Decimal


class WeeklyPlanAcceptRequest(BaseModel):
    entries: list[MealPlanEntryCreate] = Field(min_length=1, max_length=21)
    replace_existing: bool = False


class WeeklyPlanAcceptResponse(BaseModel):
    saved_count: int
    skipped_count: int