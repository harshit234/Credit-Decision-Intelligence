# Halcyon Credit — Operate Runbook
**Team Jamun | Author: Himkar | Sprint 3**

---

## 1. Service Overview

| Component | Technology | Port |
|-----------|-----------|------|
| Underwriting API | FastAPI + LangGraph | `8000` |
| Frontend Dashboard | Plain HTML/JS | `3000` |
| Vector Store | ChromaDB (local disk) | — |
| Decision Database | SQLite (`halcyon_decisions.db`) | — |
| LLM Gateway | OpenRouter (GPT-4.1) | — |

---

## 2. Service Level Objectives (SLOs)

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| End-to-end latency (p95) | < 30 seconds | > 45 seconds |
| Faithfulness score | >= 0.75 | < 0.60 |
| Cost per application | < $0.01 | > $0.015 |
| API availability | > 99% | Any 5xx burst |
| Decision accuracy (golden set) | >= 85% | < 80% |

---

## 3. Starting the Service

### Backend API
```bash
cd Credit-Decision-Intelligence
uvicorn api.main:app --reload --port 8000
```

### Frontend
```bash
python -m http.server 3000 --directory ui
```

### Check Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v2/health/detailed
```
Expected: `{"status": "ok"}` and all subsystems green.

---

## 4. Monitoring

### What to Watch in Logs

Every pipeline run prints structured JSON logs. Watch for these patterns:

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `[LLM:*] model=openai/gpt-4.1 cost=$0.00XXX` | Normal LLM call | No action |
| `[EvaluationAgent] Faithfulness=X.XXX < 0.75` | Decision failed gate | Check reasons, retry |
| `ERROR: OpenRouter API error 4XX` | Bad model ID or key issue | Check `.env` OPENROUTER_API_KEY |
| `[Router] Escalated -> human_review` | Max retries exceeded | Human underwriter must review |
| `cost=$0.0150+` per app | Above cost ceiling | Switch to CHEAP_MODEL (see below) |

### Operational Metrics Endpoint
```bash
curl http://localhost:8000/v2/metrics/operational
```
Returns: approve/decline/refer counts, avg faithfulness, avg cost.

---

## 5. Human Fallback Path

When the pipeline sets `escalated=True` in the `DecisionRecord`:

1. The `/applications` API response will show `"escalated": true` and `"recommendation": "REFER"`.
2. The frontend marks the case with an amber **"Human Review Required"** badge.
3. **Human underwriter action:**
   a. Fetch full record: `GET /v2/records/{audit_id}`
   b. Review the `trace` array for the failure reason.
   c. Review applicant profile and risk features.
   d. Override decision manually in the case management system.
   e. No re-submission through pipeline needed.

**Escalation triggers:**
- `retry_count >= 2` (LLM failed faithfulness gate twice)
- `faithfulness_score < 0.50` on final attempt

---

## 6. Cost Alarm & Model Downgrade

If average cost per application exceeds **$0.01**:

1. Open `.env`
2. Change:
   ```
   STRONG_MODEL=openai/gpt-4.1-mini
   ```
   (Switches synthesizer + evaluator to the cheaper mini model)
3. Restart the API:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. Monitor for 10 applications, confirm costs return to < $0.005/app.
5. Revert to `openai/gpt-4.1` once budget allows.

---

## 7. Rollback Procedure

If a bad deployment is detected:

```bash
# Step 1: Identify the last good commit
git log --oneline -10

# Step 2: Revert to last known good state
git revert HEAD        # creates a safe revert commit
# OR for hard rollback:
git checkout <good-commit-hash>

# Step 3: Restart API
uvicorn api.main:app --reload --port 8000

# Step 4: Run regression gate
python tests/run_regression.py
```
Only merge to `dev` if regression gate exits with code 0.

---

## 8. Incident Response Playbook

### P1 — API Down (500 errors on all requests)
1. Check uvicorn process is running.
2. Check `.env` for `OPENROUTER_API_KEY` — must start with `sk-or-`.
3. Check ChromaDB path exists: `ls chroma_db/`.
4. Check model file exists: `ls models/lgbm_halcyon_v2_lc.txt`.
5. Restart API. Run `GET /v2/health/detailed` to confirm.

### P2 — LLM Consistently Failing Faithfulness Gate
1. Check `eval/eval_results.json` for patterns.
2. If faithfulness < 0.60 on > 50% of cases: revert `gateway/prompts.py` to last known good.
3. Run `python eval/dspy_optimizer.py` to compare prompt variants.
4. Deploy winner.

### P3 — Cost Spike
1. Check `GET /v2/metrics/operational` for avg cost.
2. If avg > $0.015: switch to `STRONG_MODEL=openai/gpt-4.1-mini` immediately (see Section 6).
3. Investigate if any application is causing abnormally large context (> 2000 tokens).

### P4 — Policy Rule Regression
1. Run `python tests/run_regression.py`.
2. If any case fails: check `agents/policy_agent.py` and `tools/policy_retrieval_tool.py` for recent changes.
3. Do NOT merge the offending commit.

---

## 9. Pre-Merge Checklist (CI Gate)

Before any merge to `dev`:
- [ ] `python tests/run_regression.py` exits 0
- [ ] `python tests/integration/test_pipeline_integration.py` all pass
- [ ] `python tests/adversarial/test_adversarial.py` all pass
- [ ] No secrets in diff (`git diff | grep -i "sk-or"` returns nothing)
- [ ] `GET /v2/health/detailed` shows all subsystems `"ok"`

---

## 10. Key Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM API authentication | `sk-or-v1-...` |
| `CHEAP_MODEL` | Fast/cheap LLM path | `openai/gpt-4.1-mini` |
| `STRONG_MODEL` | High-quality LLM path | `openai/gpt-4.1` |
| `CHROMA_PERSIST_PATH` | Policy KB vector store | `./chroma_db` |
| `FAITHFULNESS_THRESHOLD` | Min score to auto-approve | `0.75` |
| `MAX_RETRIES` | LLM retry attempts | `2` |

---

*Halcyon Credit Operate Runbook — Team Jamun — Sprint 3*
