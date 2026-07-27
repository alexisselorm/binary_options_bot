from __future__ import annotations

from fastapi import APIRouter, Depends

from broker.deriv import DerivAPIWrapper
from trading.executor import TradeExecutor
from utils.config import Config
from agent.schemas import DecisionOutput, DecisionRequest
from agent.service import TradingAgentService

router = APIRouter(prefix="/agent", tags=["agent"])


def get_cfg() -> Config:
    return Config()


def get_api(cfg: Config = Depends(get_cfg)) -> DerivAPIWrapper:
    return DerivAPIWrapper(app_id=cfg.app_id)


def get_executor(
    cfg: Config = Depends(get_cfg),
    api: DerivAPIWrapper = Depends(get_api),
) -> TradeExecutor:
    return TradeExecutor(api=api, cfg=cfg)


def get_agent_service(
    cfg: Config = Depends(get_cfg),
    api: DerivAPIWrapper = Depends(get_api),
    executor: TradeExecutor = Depends(get_executor),
) -> TradingAgentService:
    return TradingAgentService(api=api, executor=executor, cfg=cfg)


@router.post("/decide", response_model=DecisionOutput)
async def decide(
    payload: DecisionRequest,
    service: TradingAgentService = Depends(get_agent_service),
) -> DecisionOutput:
    return await service.decide(payload)

