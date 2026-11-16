from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.dependencies import get_db
from features.profiles import service as profile_service
from features.savings import service
from features.savings.schemas import SavingsPeriod, SavingsSummary
from shared.auth.dependencies import get_current_user


router = APIRouter(
    prefix="/savings",
    tags=["Savings"],
)


@router.get(
    "/summary",
    response_model=SavingsSummary,
)
def get_savings_summary(
    period: SavingsPeriod = Query(default=SavingsPeriod.WEEK),
    current_user: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = profile_service.get_profile_by_auth_id(db, current_user)

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )

    return service.get_savings_summary(db, profile, period)