from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class AgentAction(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


class ProposalArgs(BaseModel):
    proposal: Literal[1] = 1
    amount: float = Field(..., gt=0)
    basis: Literal["stake"] = "stake"
    contract_type: Literal["CALL", "PUT"]
    currency: Literal["USD"] = "USD"
    duration: int = Field(..., gt=0)
    duration_unit: Literal["s"] = "s"
    symbol: str = Field(..., min_length=1)


class DecisionRequest(BaseModel):
    symbol: str
    timeframe_minutes: int = Field(..., gt=0)
    strategy: str
    candle_count: int = Field(default=2000, gt=0)
    expiry_seconds: int = Field(default=60, gt=0)


class DecisionOutput(BaseModel):
    action: AgentAction
    proposal_args: Optional[ProposalArgs] = None
    signal: Optional[Literal["CALL", "PUT"]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1, max_length=400)
    tools_used: List[str] = Field(default_factory=list)
    dry_run: bool = True
    risk_checks: Dict[str, Any] = Field(default_factory=dict)
    trade_receipt: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trade_fields(self):
        if self.action == AgentAction.TRADE and not self.proposal_args:
            raise ValueError("proposal_args required when action is TRADE")
        if self.action == AgentAction.NO_TRADE and self.proposal_args is not None:
            raise ValueError("proposal_args must be null when action is NO_TRADE")
        return self

