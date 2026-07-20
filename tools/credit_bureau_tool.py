"""
================================================================================
   HALCYON CREDIT — Credit Bureau Tool (Mock)
   Stage 3 | Author: Aditya
   Simulates a credit bureau API pull (Equifax / CIBIL / Experian).
   Returns structured bureau data from applicant-provided fields + adjustments.
   Production: replace with real bureau API call + discrepancy detection.
================================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BureauRecord:
    applicant_id:      str
    credit_score:      int
    delinquencies:     int
    credit_age_months: int
    open_accounts:     int
    utilization_pct:   float
    thin_file:         bool
    bureau_ref_id:     str
    fetched_at:        str


def fetch_bureau(applicant_id: str, applicant_data: dict) -> BureauRecord:
    """
    Fetch credit bureau data for an applicant.

    In mock mode: uses applicant-provided fields + small conservative adjustments
    to simulate what a real bureau pull would return.

    In production: this calls Equifax / CIBIL API and cross-checks stated values
    against actual bureau data, flagging discrepancies.

    Args:
        applicant_id:   Unique applicant ID
        applicant_data: Dict with applicant credit fields from ApplicantFile

    Returns:
        BureauRecord with validated credit data
    """
    credit_score      = int(applicant_data.get("credit_score", 650))
    delinquencies     = int(applicant_data.get("delinquencies_2yr", 0))
    credit_age_months = int(applicant_data.get("credit_age_months", 60))
    open_accounts     = int(applicant_data.get("open_accounts", 5))
    utilization_pct   = float(applicant_data.get("revolving_utilisation", 30.0))
    verification      = applicant_data.get("verification_status", "Not Verified")

    # Conservative bureau adjustment for unverified applicants
    # Simulates the discrepancy detection a real bureau would flag
    if verification == "Not Verified":
        credit_score = max(300, credit_score - 15)   # haircut unverified score

    # Thin file: less than 2 years of credit history OR fewer than 3 open accounts
    thin_file = (credit_age_months < 24) or (open_accounts < 3)

    return BureauRecord(
        applicant_id      = applicant_id,
        credit_score      = credit_score,
        delinquencies     = delinquencies,
        credit_age_months = credit_age_months,
        open_accounts     = open_accounts,
        utilization_pct   = utilization_pct,
        thin_file         = thin_file,
        bureau_ref_id     = f"BUR-{applicant_id}-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
        fetched_at        = datetime.utcnow().isoformat(),
    )
