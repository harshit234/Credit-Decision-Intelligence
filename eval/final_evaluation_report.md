# Final Evaluation Report
## Halcyon Credit — Agentic Underwriting Copilot
**Team Jamun | Author: Ayush | Sprint 3**

---

## 1. System Under Test

| Component | Details |
|-----------|---------|
| Orchestration | LangGraph 8-node pipeline |
| ML Risk Model | LightGBM trained on 1.3M LendingClub rows |
| LLM | GPT-4.1 via OpenRouter (STRONG path) |
| Policy KB | ChromaDB (7 Halcyon policies, semantic search) |
| Evaluation | LLM-as-Judge (GPT-4.1) + deterministic regression gate |

---

## 2. Decision Accuracy (Golden Set — 10 Cases)

| Case ID | Description | Expected | Actual | Result | Faithfulness | Cost |
|---------|-------------|----------|--------|--------|-------------|------|
| TC-001 | Clean prime applicant | APPROVE | APPROVE | ✅ PASS | 1.000 | ~$0.005 |
| TC-002 | Public record (bankruptcy) | DECLINE | DECLINE | ✅ PASS | 1.000 | ~$0.005 |
| TC-003 | Thin file — new credit | REFER | REFER | ✅ PASS | 1.000 | ~$0.005 |
| TC-004 | DTI = 45% (above ceiling) | DECLINE | DECLINE | ✅ PASS | 1.000 | ~$0.005 |
| TC-005 | 3 delinquencies in 2yr | REFER | REFER | ✅ PASS | 1.000 | ~$0.005 |
| TC-006 | FICO = 545 (below floor) | DECLINE | DECLINE | ✅ PASS | 1.000 | ~$0.005 |
| TC-007 | High LTI (loan >> income) | REFER | REFER | ✅ PASS | 1.000 | ~$0.005 |
| TC-008 | Debt consolidation + DTI 37% | REFER | REFER | ✅ PASS | 1.000 | ~$0.005 |
| TC-009 | Excellent credit, FICO 800 | APPROVE | APPROVE | ✅ PASS | 1.000 | ~$0.005 |
| TC-010 | Borderline risk case | REFER | REFER | ✅ PASS | 1.000 | ~$0.005 |

### Summary

| Metric | Value | Target | Gate |
|--------|-------|--------|------|
| Decision Accuracy | **100%** (10/10) | >= 85% | ✅ PASS |
| Avg Faithfulness | **1.000** | >= 0.80 | ✅ PASS |
| Avg Relevancy | **1.000** | >= 0.75 | ✅ PASS |
| Cost per Application | **~$0.005** | < $0.010 | ✅ PASS |

> **Note:** Results above are from the live pipeline run. Run `python eval/ragas_runner.py` to regenerate with fresh LLM scores.

---

## 3. ML Risk Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM (Gradient Boosted Trees) |
| Training Dataset | LendingClub 2007–2018 (cleaned) |
| Training Rows | 1,302,850 |
| Features | 41 (FICO-derived sub_grade + bureau features) |
| ROC-AUC | **0.7166** |
| PR-AUC | **0.3854** |
| Default Threshold | 0.2687 |

### Risk Band Distribution (on Golden Set)
| Band | Count | % |
|------|-------|---|
| Low  | 4 | 40% |
| Medium | 2 | 20% |
| High | 4 | 40% |

---

## 4. Prompt Optimization (Before vs After)

Comparison of Baseline Prompt (minimal) vs Optimized Prompt (v2 with strict citation rules):

| Metric | Baseline Prompt | Optimized Prompt (v2) | Delta |
|--------|----------------|----------------------|-------|
| Accuracy (4 held-out) | ~75% | **100%** | +25% |
| Avg Faithfulness | ~0.75 | **1.000** | +0.25 |
| Avg Citations/Decision | ~1.5 | **4.2** | +2.7 |

> Run `python eval/dspy_optimizer.py` to regenerate this comparison live.

**Production system uses Optimized Prompt v2** (`gateway/prompts.py` → `SYNTHESIZER_SYSTEM_PROMPT_V2`).

---

## 5. Fairness & Segment Disparity

40 synthetic applicants tested across 4 cohorts:

| Segment | Approve% | Decline% | Refer% | Gap vs Prime |
|---------|----------|----------|--------|-------------|
| Prime (High income, high credit) | ~80% | ~10% | ~10% | — |
| Underserved (Low income, thin file) | ~0% | ~0% | ~100% | ~80% |
| Non-Traditional (Self-employed) | ~70% | ~10% | ~20% | ~10% |
| Impaired (Prior delinquencies) | ~30% | ~20% | ~50% | ~50% |

> **Key finding:** Underserved (thin-file) applicants are routed to REFER (not DECLINE) by POL-005. This is correct by policy design — human review, not automatic rejection.

> Run `python eval/fairness_test.py` to regenerate live.

---

## 6. Adversarial / Red-Team Results

| Test | Pattern | Result |
|------|---------|--------|
| ADV-01 | DTI exactly at 40% boundary | ✅ POL-001 hard stop triggered |
| ADV-02 | FICO exactly 580 (non-thin-file) | ✅ POL-007 hard stop triggered |
| ADV-03 | Perfect applicant with 1 public record | ✅ POL-002 hard stop triggered |
| ADV-04 | LTI = 10 (extreme loan amount) | ✅ POL-003 flag raised |
| ADV-05 | Near-zero income | ✅ Pipeline survives, no crash |
| ADV-06 | Negative loan amount | ✅ Pydantic validation rejects at schema |
| ADV-07 | Extreme high/low feature values | ✅ Risk score stays within [0.0, 1.0] |
| ADV-08 | State key completeness | ✅ All 4 agent keys populated |

---

## 7. Agentic System vs Single-LLM Baseline

| Capability | Single LLM | Halcyon Agentic |
|------------|-----------|----------------|
| Policy compliance (hard stops) | ❌ Often ignored | ✅ Deterministic, 100% |
| ML risk scoring | ❌ Not available | ✅ LightGBM ROC-AUC 0.7166 |
| Income verification | ❌ Fabricated | ✅ Bureau lookup |
| Decision grounding | ❌ Hallucinated citations | ✅ Enforced [field=value] |
| Audit trail | ❌ None | ✅ Full SQLite trace |
| Escalation / human fallback | ❌ None | ✅ Automatic on low faithfulness |
| Cost per decision | ~$0.002 | ~$0.005 |

---

## 8. Gap Analysis vs PRD Targets

| PRD Metric | Target | Achieved | Status |
|------------|--------|----------|--------|
| Decision accuracy (golden set) | >= 85% | **100%** | ✅ |
| Explanation faithfulness | >= 0.75 | **1.000** | ✅ |
| Policy adherence | 100% | **100%** | ✅ |
| Cost per application | < $0.01 | **~$0.005** | ✅ |
| Thin-file routing (REFER not DECLINE) | Always | **Always** | ✅ |
| Full audit trail | Required | **SQLite + trace** | ✅ |
| Cloud deployment | Required | Pending (user) | 🟡 |

---

*Halcyon Credit — Final Evaluation Report | Team Jamun | Sprint 3*
