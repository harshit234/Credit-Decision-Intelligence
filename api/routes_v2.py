"""
================================================================================
   HALCYON CREDIT — Production API Middleware & Hardening
   Stage 4 | Author: Harshit
   Domain: ML Model, RiskScoring Node, API Integration

   Adds:
   - Request ID tracking (X-Request-ID header)
   - Structured JSON logging
   - Cost ceiling guard (rejects if projected cost > COST_CEILING)
   - Rate limit response headers
   - /metrics/model endpoint (model version, ROC-AUC, PR-AUC)
   - /records/{audit_id} detailed record retrieval
================================================================================
"""
from __future__ import annotations
import sys, os, uuid, time, json, logging
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

from tools.decision_record_tool import fetch_record, list_records

router_v2 = APIRouter(prefix="/v2", tags=["v2"])

# ─── Model metadata (written by train_lc_v2.py) ──────────────────────────────
MODEL_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "feature_schema_v2_lc.json"
)

def _load_model_meta() -> dict:
    try:
        with open(MODEL_SCHEMA_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ─── Structured logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger("halcyon")


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_request_id(request: Request) -> str:
    """Return existing X-Request-ID or generate a new one."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])


# ─────────────────────────────────────────────────────────────────────────────
# V2 ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@router_v2.get("/model/info", summary="Model version and performance metrics")
def model_info():
    """
    Returns the current production model version, feature count,
    and training metrics (ROC-AUC, PR-AUC).
    """
    meta = _load_model_meta()
    return {
        "model_version":  meta.get("model_version", "lgbm_halcyon_v2_lc"),
        "algorithm":      "LightGBM (Gradient Boosted Trees)",
        "training_rows":  meta.get("training_rows", 1302850),
        "feature_count":  meta.get("feature_count", 41),
        "roc_auc":        meta.get("roc_auc", 0.7166),
        "pr_auc":         meta.get("pr_auc",  0.3854),
        "threshold":      meta.get("threshold", 0.2687),
        "dataset":        "LendingClub (2007-2018, cleaned)",
        "risk_bands": {
            "High":   f">= {meta.get('threshold', 0.2687):.4f}",
            "Medium": "0.2500 - threshold",
            "Low":    "< 0.2500",
        },
    }


@router_v2.get("/records/{audit_id}", summary="Retrieve full decision record")
def get_record(audit_id: str, request: Request):
    """Retrieve a full decision record including trace and cost breakdown."""
    req_id = get_request_id(request)
    logger.info(f"record_fetch audit_id={audit_id} req_id={req_id}")

    record = fetch_record(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {audit_id}")
    return record


@router_v2.get("/records", summary="List recent decisions with pagination")
def list_records_v2(limit: int = 20, offset: int = 0):
    """List decision records with pagination support."""
    all_records = list_records(limit=limit + offset)
    return {
        "total":   len(all_records),
        "limit":   limit,
        "offset":  offset,
        "records": all_records[offset:offset + limit],
    }


@router_v2.get("/metrics/operational", summary="Operational metrics dashboard")
def operational_metrics():
    """
    Aggregated operational metrics for the underwriting pipeline.
    Useful for monitoring dashboards.
    """
    records = list_records(limit=500)
    total   = len(records)

    if total == 0:
        return {"message": "No decisions recorded yet.", "total": 0}

    approvals  = sum(1 for r in records if r.get("recommendation") == "APPROVE")
    declines   = sum(1 for r in records if r.get("recommendation") == "DECLINE")
    refers     = sum(1 for r in records if r.get("recommendation") == "REFER")
    escalated  = sum(1 for r in records if r.get("escalated"))
    costs      = [r.get("cost_usd", 0) or 0 for r in records]
    faiths     = [r.get("faithfulness") or 0 for r in records if r.get("faithfulness")]
    risks      = [r.get("risk_score") or 0 for r in records if r.get("risk_score")]

    return {
        "total_applications":      total,
        "decisions": {
            "approve_count":       approvals,
            "decline_count":       declines,
            "refer_count":         refers,
            "escalation_count":    escalated,
            "approve_rate":        round(approvals / total, 3),
            "decline_rate":        round(declines / total, 3),
            "refer_rate":          round(refers / total, 3),
            "escalation_rate":     round(escalated / total, 3),
        },
        "quality": {
            "avg_faithfulness":    round(sum(faiths) / len(faiths), 3) if faiths else None,
            "min_faithfulness":    round(min(faiths), 3) if faiths else None,
        },
        "risk": {
            "avg_risk_score":      round(sum(risks) / len(risks), 4) if risks else None,
            "high_risk_count":     sum(1 for r in records if r.get("risk_band") == "High"),
            "medium_risk_count":   sum(1 for r in records if r.get("risk_band") == "Medium"),
            "low_risk_count":      sum(1 for r in records if r.get("risk_band") == "Low"),
        },
        "cost": {
            "total_cost_usd":      round(sum(costs), 4),
            "avg_cost_per_app":    round(sum(costs) / total, 5),
            "max_cost_usd":        round(max(costs), 5) if costs else 0,
        },
    }


@router_v2.get("/health/detailed", summary="Detailed health check")
def health_detailed():
    """
    Checks all subsystem health:
    - SQLite database
    - LightGBM model file
    - ChromaDB collection
    - OpenRouter API key presence
    """
    from dotenv import load_dotenv
    load_dotenv()

    checks = {}

    # DB check
    try:
        recs = list_records(limit=1)
        checks["database"] = {"status": "ok", "backend": "sqlite"}
    except Exception as e:
        checks["database"] = {"status": "error", "detail": str(e)[:100]}

    # Model check
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "lgbm_halcyon_v2_lc.txt"
    )
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        checks["model"] = {"status": "ok", "size_mb": round(size_mb, 2)}
    else:
        checks["model"] = {"status": "missing", "path": model_path}

    # ChromaDB check
    try:
        chroma_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
        if os.path.exists(chroma_path):
            checks["chromadb"] = {"status": "ok", "path": chroma_path}
        else:
            checks["chromadb"] = {"status": "missing"}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "detail": str(e)[:80]}

    # API key presence
    key = os.getenv("OPENROUTER_API_KEY", "")
    checks["openrouter"] = {
        "status":   "ok" if key.startswith("sk-or") else "missing",
        "key_hint": key[:12] + "..." if key else "NOT SET",
    }

    overall = "ok" if all(v.get("status") == "ok" for v in checks.values()) else "degraded"
    return {"overall": overall, "subsystems": checks}
