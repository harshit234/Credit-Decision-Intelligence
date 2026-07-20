"""
================================================================================
   HALCYON CREDIT — FastAPI Application Entry Point
   Stage 3 | Author: Himkar
   Run with: uvicorn api.main:app --reload --port 8000
================================================================================
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.routes_v2 import router_v2

app = FastAPI(
    title       = "Halcyon Credit — Agentic Underwriting Copilot",
    description = (
        "Multi-agent LangGraph pipeline for automated loan underwriting. "
        "Team Jamun | Futurense × IIT Gandhinagar | Cohort 1 Capstone"
    ),
    version     = "3.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allow all in dev; restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router,    prefix="")
app.include_router(router_v2, prefix="")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
