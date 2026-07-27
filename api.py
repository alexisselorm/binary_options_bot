from __future__ import annotations

from fastapi import FastAPI
from agent.router import router as agent_router

app = FastAPI(title="Binary Options Bot API")
app.include_router(agent_router)

