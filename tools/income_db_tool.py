"""
================================================================================
   HALCYON CREDIT — Income DB Tool (Mock)
   Stage 3 | Author: Aditya
   Simulates an income verification database lookup.
   Production: replace with Payroll / ITR / Bank-statement API.
================================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IncomeRecord:
    applicant_id:    str
    reported_income: float
    verified_income: float
    confidence:      float   # 0.0–1.0
    source:          str     # W2 | bank_statement | self_reported | derived
    fetched_at:      str


# Mock in-memory income database (keyed by applicant_id)
_INCOME_DB: dict[str, dict] = {
    "APL-001": {"reported": 85000,  "verified": 76500,  "confidence": 0.90, "source": "W2"},
    "APL-002": {"reported": 45000,  "verified": 33750,  "confidence": 0.75, "source": "bank_statement"},
    "APL-003": {"reported": 120000, "verified": 108000, "confidence": 0.90, "source": "W2"},
    "APL-004": {"reported": 32000,  "verified": 12800,  "confidence": 0.40, "source": "self_reported"},
    "APL-005": {"reported": 65000,  "verified": 58500,  "confidence": 0.90, "source": "W2"},
    "APL-006": {"reported": 28000,  "verified": 21000,  "confidence": 0.75, "source": "bank_statement"},
    "APL-007": {"reported": 95000,  "verified": 85500,  "confidence": 0.90, "source": "W2"},
    "APL-008": {"reported": 55000,  "verified": 22000,  "confidence": 0.40, "source": "self_reported"},
    "APL-009": {"reported": 150000, "verified": 135000, "confidence": 0.90, "source": "W2"},
    "APL-010": {"reported": 38000,  "verified": 28500,  "confidence": 0.75, "source": "bank_statement"},
}

# Confidence map based on verification status
_CONFIDENCE_MAP = {
    "Source Verified": 0.90,
    "Verified":        0.75,
    "Not Verified":    0.40,
}


def lookup_income(
    applicant_id:        str,
    stated_income:       float,
    verification_status: str,
) -> IncomeRecord:
    """
    Look up verified income for an applicant.
    Falls back to stated-income × confidence when applicant not in mock DB.

    Args:
        applicant_id:        Unique applicant identifier
        stated_income:       Income declared by applicant on the form
        verification_status: Source Verified | Verified | Not Verified

    Returns:
        IncomeRecord with verified income and confidence score
    """
    if applicant_id in _INCOME_DB:
        r = _INCOME_DB[applicant_id]
        return IncomeRecord(
            applicant_id    = applicant_id,
            reported_income = r["reported"],
            verified_income = r["verified"],
            confidence      = r["confidence"],
            source          = r["source"],
            fetched_at      = datetime.utcnow().isoformat(),
        )

    # Unknown applicant — derive from verification status
    confidence      = _CONFIDENCE_MAP.get(verification_status, 0.40)
    verified_income = round(stated_income * confidence, 2)

    return IncomeRecord(
        applicant_id    = applicant_id,
        reported_income = stated_income,
        verified_income = verified_income,
        confidence      = confidence,
        source          = "derived_from_verification_status",
        fetched_at      = datetime.utcnow().isoformat(),
    )
