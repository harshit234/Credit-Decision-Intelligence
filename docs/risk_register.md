# Risk Register — Halcyon Credit Agentic Underwriting Copilot

> Living document. Likelihood × Impact drives priority. Every risk has an owner and a mitigation.
> Update when a risk materialises, is mitigated, or a new one appears. Mirrors PRD §6 and expands it.

**Scale:** Likelihood / Impact = Low · Medium · High. **Status:** Open · Mitigating · Accepted · Closed.

---

## 1. Model & decision-quality risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-01 | Hallucinated income / credit facts in the written decision | M | High | Ayush | `source_refs` required on all worker outputs; Evaluation Agent checks every claim against state; faithfulness gate | Mitigating |
| R-02 | Unexplainable decline (no defensible reason) | L | High | Harshit | SHAP top-5 mandatory in risk output; reasons must cite source fields; faithfulness gate | Mitigating |
| R-03 | Risk model poorly calibrated (scores ≠ real default probability) | M | High | Harshit | Calibration curve at validation; report Brier/ECE; threshold tuned on holdout | Open |
| R-04 | Uncalibrated LLM-as-Judge gives false confidence | M | High | Ayush | Calibrate judge vs human-labelled set; report agreement (κ) before trusting scores | Open |

## 2. Fairness & ethics risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-05 | Protected-class bias amplified by the model | M | High | shared | Fairness suite every eval run; segment gap alert > 5pp; protected attrs never model inputs | Open |
| R-06 | Proxy variables leak protected attributes | M | Med | shared | Document proxy methodology; correlation audit of features vs protected attrs | Open |
| R-07 | Over-automation removes human judgment | L | High | Aditya | Copilot framing (recommendation, not binding); human escalation forced on retry exhaustion | Mitigating |

## 3. Retrieval & policy risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-08 | Policy retrieval miss (relevant clause not retrieved) | M | Med | Ayush | Fallback to conservative defaults + alert + human escalation; context-recall metric tracked | Open |
| R-09 | Policy corpus stale / wrong jurisdiction | L | Med | Ayush | Clause metadata (effective_date, jurisdiction); versioned corpus | Open |

## 4. Security & privacy risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-10 | Applicant PII exposed via prompt logs | L | High | Himkar | PII scrubbing before LiteLLM call; `applicant_id` reference only; log audit | Mitigating |
| R-11 | Secrets committed to the repo | L | High | all | `.env` gitignored; secret scan in CI; rotate-on-exposure policy | Mitigating |

## 5. Operational & cost risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-12 | Cost overrun per application | M | Med | Himkar | Token budget; LiteLLM cost alarms; cheap-path routing for non-critical agents | Open |
| R-13 | Latency exceeds 30s p95 | M | Med | Aditya | Async fan-out; semantic caching of policy lookups; load test in Sprint 3 | Open |
| R-14 | LLM provider outage / rate limit | M | Med | Himkar | LiteLLM fallback chain; retries with backoff; degrade to human queue | Mitigating |

## 6. Project & delivery risks

| ID | Risk | L | I | Owner | Mitigation | Status |
|---|---|:--:|:--:|---|---|---|
| R-15 | Half the grade (Eng + Eval) unstarted vs deadline | H | High | all | Front-load vertical slice + golden set + baseline; weekly progress review | Open |
| R-16 | Uneven individual contribution → viva-gate failure | M | High | all | Per-member owned verticals; contribution log in PROGRESS §9; cross-reviewed PRs | Open |
| R-17 | Dataset undecided/unprofiled blocks model + golden set | M | High | Harshit | Ratify Home Credit (ADR-006); EDA + dataset card this sprint | Open |

---

### Top risks to watch this sprint
**R-15** (schedule), **R-16** (viva contribution), **R-17** (dataset), **R-04** (judge calibration). These gate the highest-weight rubric buckets.
