from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.dependencies import get_db
from shared.auth.dependencies import get_current_user

from features.profiles import service as profile_service
from features.weekly_plan import service
from features.weekly_plan.schemas import (
    WeeklyPlanAcceptRequest,
    WeeklyPlanAcceptResponse,
    WeeklyPlanDraftResponse,
    WeeklyPlanGenerateRequest,
)


router = APIRouter(
    prefix="/weekly-plan",
    tags=["Weekly Plan"],
)


def _get_profile_or_404(db: Session, current_user: UUID):
    profile = profile_service.get_profile_by_auth_id(db, current_user)

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )

    return profile


@router.post(
    "/generate",
    response_model=WeeklyPlanDraftResponse,
)
def generate_weekly_plan(
    request: WeeklyPlanGenerateRequest,
    current_user: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = _get_profile_or_404(db, current_user)

    return service.generate_weekly_plan(db, profile.id, request)


@router.post(
    "/accept",
    response_model=WeeklyPlanAcceptResponse,
    status_code=status.HTTP_201_CREATED,
)
def accept_weekly_plan(
    request: WeeklyPlanAcceptRequest,
    current_user: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = _get_profile_or_404(db, current_user)

    return service.accept_weekly_plan(db, profile.id, request)