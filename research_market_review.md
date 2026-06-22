# Market Research & Prior Art Review
## Halcyon Credit — Agentic Underwriting Copilot

**Team Jamun** · Harshit · Ayush · Aditya · Himkar  
**Program:** Futurense × IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1  
**Capstone Project 02** · Version 1.0 · June 2025

---

## 1. Purpose

This document surveys three deployed, production-grade systems that operate in the same problem space as the Halcyon Credit Agentic Underwriting Copilot. Each one illuminated a different aspect of the challenge — thin-file lending at scale, explainability and fairness under regulatory scrutiny, and multi-agent orchestration with immutable audit trails — and collectively gave Team Jamun the evidence and conceptual vocabulary to design our solution. The research is presented not as a literature review but as an honest account of what we observed, what impressed us, what we found lacking, and what we resolved to do differently.

---

## 2. Solution 1 — Upstart: AI Lending for Thin-File Borrowers at Scale

### 2.1 What It Is

Upstart is an AI-native lending platform founded in 2012, now processing personal, auto, and home equity loans for more than 500 bank and credit union partners across the United States. Its core thesis is that FICO-based underwriting systematically excludes creditworthy borrowers — particularly those with thin credit files, recent immigrants, and young adults — because it relies on only 15–20 credit bureau variables. Upstart replaces this narrow scorecard with a machine learning model trained on more than 2,500 variables and 100+ million monthly repayment events.

### 2.2 How It Works

When a borrower submits an application, Upstart's model runs within seconds and produces a credit decision — approve, decline, or refer — along with a suggested interest rate. The model draws on income and employment signals, education history, repayment behaviour, and macroeconomic context (via Upstart's proprietary Macro Index, UMI) in addition to standard bureau data. For thin-file applicants, alternative verification pathways kick in: in Q4 2024, 82% of thin-file applications were verified using non-traditional signals such as job title, employer stability, and educational background. The system is deeply automated — as of early 2026, more than 91% of loans are processed without any human involvement by Upstart's team, with lenders retaining full control over their credit policy thresholds.

### 2.3 Measured Outcomes

The published performance numbers are independently auditable and striking. Against a hypothetical traditional model run on the same data:

- **44% more borrowers approved** at the same loss rate
- **36% lower APRs** offered at the same approval rates
- **116% more Black borrowers and 123% more Hispanic borrowers** approved at lower APRs versus traditional models (Q4 2024 figures)
- **$11 billion in originations** in fiscal year 2025, up 86% year-over-year
- **89% accuracy** predicting 90-day defaults, versus 72% for FICO-only models

### 2.4 What We Learned

Upstart's approach validated two foundational assumptions behind our design. First, thin-file applicants are not inherently high risk — they are *unseen* by narrow data models, and richer features genuinely improve both risk separation and fairness simultaneously. This is the same population Halcyon Credit serves, and it gave us confidence that a well-trained ML risk model (our XGBoost / LightGBM layer) is a legitimate and measurable improvement over scorecard-based alternatives.

Second, 91% automation does not mean 91% of decisions are unaccountable. Upstart makes clear that every lender retains policy control and that declined applicants receive written reasons under adverse action requirements. This confirmed for us that automation and explainability are not in tension — but the *mechanism* for producing explainable reasons is where Upstart leaves a gap.

### 2.5 The Gap That Motivated Us

Upstart's ML model produces a score. It does not produce a narrative. The reasons given to declined applicants are generated from score-feature attributions, not from a reasoning chain that checks policy compliance, cross-references bureau data against stated income, and synthesizes a written explanation that a compliance officer could review and a regulator could interrogate. For a lender like Halcyon operating without Upstart's legal and compliance team, this distinction matters enormously. A score with attribution codes is not the same as an auditable decision record with cited evidence. We designed the Decision Synthesizer Agent and the Evaluation Agent precisely to fill this gap — to produce not just a risk score but a full reasoning chain that is grounded, faithful, and policy-compliant.

---

## 3. Solution 2 — Zest AI: Explainable Machine Learning Underwriting with Fairness Controls

### 3.1 What It Is

Zest AI is an enterprise AI underwriting platform founded in 2009 and deployed by more than 175 lenders ranging from the largest US banks to small community credit unions. Its core product is a Model Management System that allows lenders to build, validate, deploy, and monitor custom ML credit models — gradient-boosted and ensemble models trained on each institution's own historical loan performance data. As of 2024, Zest AI has built and deployed more than 500 active proprietary credit models and secured over 50 patents, and received a $200 million growth investment from Insight Partners in late 2024. It is widely regarded as the pioneer of fairness-optimised, explainable ML underwriting in US consumer lending.

### 3.2 How It Works

Unlike Upstart, which operates as a marketplace where Upstart's model makes the decision, Zest AI gives lenders their own custom models trained on their own portfolios. The platform handles the full ML model lifecycle: data ingestion, feature engineering, model training, validation against holdout samples, deployment, and ongoing monitoring. Crucially, Zest AI built explainability in from the start rather than bolting it on. Every model output includes a mathematically rigorous contribution score for each input feature — using game-theoretic methods (analogous to SHAP values) — so that the principal reasons for any individual decision can be precisely stated. This is what regulators and the CFPB require for adverse action notices, and Zest AI generates the compliance documentation automatically.

On fairness, Zest AI runs disparate impact analysis across protected classes at every model validation step. The fairness optimisation is not a post-hoc adjustment but is built into the model training objective: the goal is simultaneously better risk separation and better fairness. Published results show that Zest AI models have increased approvals by 49% for Latino applicants, 41% for Black applicants, and 40% for women — all at the same loss rate, meaning fairness improved without adding risk.

### 3.3 Measured Outcomes

- **20–25% more loan approvals** without added risk for lender portfolios
- **Up to 20% lower default rates** at constant approval rates
- **Auto-decisioning rates of 60–80%+** achieved by credit union customers
- **100% retention rate** among credit union customers, NPS of 81 (top 1% across all industries)
- **39 million applications assessed** over four years (through 2023), resulting in $250 billion in loans granted

### 3.4 What We Learned

Zest AI gave us our most important design constraint: **explainability must be mathematically grounded, not narratively improvised**. The platform demonstrated that SHAP-value-style feature attributions are not a "nice to have" — they are the mechanism by which a declined applicant's adverse action notice is legally defensible. Every reason code traces back to a specific input value with a measured contribution to the model output. We adopted this directly: our Risk Scoring Agent is required to output SHAP top-5 feature attributions alongside the risk score and risk band, and these values are passed to the Decision Synthesizer as structured inputs so that the written decision reasons are grounded in the same evidence the model used.

Zest AI also showed us what enterprise-grade fairness testing looks like in practice: not a checkbox but a continuous measurement process with segment-level approval rate comparisons run at every model validation cycle. Our fairness testing module in Sprint 3 mirrors this structure — segment gap alerts, approval rate differentials across employment type and income band, and a fairness gate on every golden-set evaluation run.

### 3.5 The Gap That Motivated Us

Zest AI is a model management platform. It does not orchestrate a multi-agent reasoning pipeline. It produces a score with explanations, but it does not check that score against a policy knowledge base, synthesize a narrative recommendation, evaluate the faithfulness of that narrative against the underlying data, or persist a full execution trace for audit replay. For a lender without a dedicated compliance team, these steps still require human underwriters — which is precisely what Halcyon Credit cannot afford at scale. Our Orchestrator-Worker + Evaluator-Optimizer architecture was designed to automate the *full* underwriting workflow that Zest AI's score feeds into, not just the scoring step.

Additionally, Zest AI's solution is an enterprise product with six-figure annual contracts and multi-year agreements, requiring the lender to bring clean historical loan data and integration engineering. It is not accessible to a digital consumer lender still building its data infrastructure. Halcyon Credit needed a solution it could own, build, and audit internally — which pointed us toward an open-stack architecture (LangGraph, ChromaDB, XGBoost, FastAPI) rather than a proprietary platform.

---

## 4. Solution 3 — Agentic Process Automation in Lending (Automation Anywhere + AWS Bedrock Patterns)

### 4.1 What It Is

The third influence on our design is not a single company but a convergent pattern that emerged across multiple production deployments in 2024–2025: the application of **multi-agent orchestration** to regulated financial workflows. Two implementations are particularly instructive.

**Automation Anywhere** — a Gartner Magic Quadrant Leader for RPA seven consecutive years — launched an Agentic Process Automation (APA) platform that orchestrates AI agents, RPA bots, APIs, and human review gates in a single workflow. Their loan underwriting agentic deployment reported a **60% reduction in processing time**, with one automotive lender achieving an **88% reduction in approval cycles**. The platform uses four actor types: AI reasoning agents, deterministic RPA bots for structured tasks, APIs for system-to-system data exchange, and mandatory human review gates for high-stakes decisions.

**AWS Bedrock AgentCore** — Amazon's managed multi-agent framework — published a reference architecture in late 2025 for intelligent loan application processing using a **graph pattern**: a supervisor agent (loan underwriting supervisor) coordinates a hierarchy of specialized sub-agents (financial analysis, risk analysis, credit assessment, verification, fraud detection, policy documentation), each with focused, narrow responsibilities and tool access. The key design insight documented by AWS was the **anti-pattern of agent overloading** — a single agent given too many tools and too broad a scope produces confused tool selection, incorrect arguments, and inconsistent responses at production scale.

### 4.2 How It Works

Both implementations share a common structural pattern: a central orchestrator decomposes the underwriting task into parallel specialist workstreams, collects and merges results, then passes the combined output to downstream synthesis and evaluation layers. Governance is embedded throughout — not added after. Every state transition, data handoff, and agent action is immutably logged. Human escalation is a designed pathway, not a failure fallback.

The Datamatics insurance underwriting case study (using CrewAI as the orchestration layer) is also directly instructive: it deployed four specialized agents — Risk Assessor, Preliminary Outcome Evaluator, Underwriter Decision Maker, and Final Evaluator — in a pipeline where the Final Evaluator acts as a quality gate before output is committed. Every agent produces structured, auditable JSON. The final decision explicitly cites the evidence it rests on ("High risk due to uncontrolled hypertension and diabetes history"). This is, almost exactly, the architecture we designed independently before finding this case study — which served as confirming evidence that the pattern is viable in production.

### 4.3 Measured Outcomes

- **Automation Anywhere loan underwriting:** 60% reduction in processing time; 88% reduction in approval cycles (automotive lender)
- **AWS Bedrock graph-pattern underwriting:** reduced manual underwriting time with full audit trail, documented in production-grade reference architecture
- **Agentic AI lending broadly (McKinsey, 2024):** AI-driven credit models analysing up to 10,000 data points versus 50–100 in traditional scoring; 30–40% reduction in per-loan processing costs at leading institutions
- **Lenders using AI-based underwriting (Freddie Mac, 2024):** 14% reduction in per-loan origination costs, 40% fewer defects, 5-day shorter loan production cycles

### 4.4 What We Learned

The most important lesson from these deployments is the **agent responsibility principle**: each agent must have a narrow, well-defined scope with a small number of tools. The AWS anti-pattern documentation was particularly clarifying — an agent with too many responsibilities starts calling wrong tools, passing bad arguments, and producing inconsistent output. This directly shaped our contract design: each of our seven agents reads from and writes to specific named state keys, with no agent allowed to touch state it does not own. The policy is enforced by Pydantic validators on the LangGraph node wrappers, not by convention.

The Automation Anywhere architecture showed us that the four-actor model — reasoning agents, deterministic bots, APIs, human gates — maps well onto the LangGraph parallel fan-out pattern we were already considering. The deterministic data-retrieval steps (Income DB, Credit Bureau) are the equivalent of RPA bots in their architecture: they should bypass LLM reasoning entirely and use direct tool calls, keeping cost low and reliability high. The LLM reasoning is reserved for steps that genuinely require it: policy clause relevance ranking, decision synthesis, and evaluation.

The Datamatics insurance case study gave us confidence in the Evaluator-Optimizer loop. An evaluation agent that reads the draft decision alongside the source data, identifies unsupported claims, and either passes or routes back to the synthesizer is a proven production pattern in adjacent regulated domains. We extended it by making the retry loop stateful (judge feedback is appended to state on every pass), capped (max N=2 retries), and escalation-gated (human review queue on exhaustion).

### 4.5 The Gap That Motivated Us

Enterprise agentic platforms like Automation Anywhere are powerful but carry the same accessibility constraint as Zest AI: they are designed for large institutions with existing automation investments, dedicated engineering teams, and six-figure contract budgets. The open-source and cloud-native ecosystem — LangGraph, ChromaDB, LiteLLM, FastAPI — now offers most of the same capabilities with full ownership, full auditability, and no vendor lock-in. For an early-stage digital lender like Halcyon Credit, that matters.

More importantly, none of the surveyed deployments combined all four elements we needed in a single system: (1) ML risk scoring with SHAP attribution, (2) RAG-grounded policy compliance checking, (3) LLM-as-judge faithfulness evaluation with a stateful retry loop, and (4) a fully replayable, immutable audit trail at the state-graph level. Each existing solution addresses one or two of these well. Halcyon Credit needed all four together — which is why we are building the Agentic Underwriting Copilot from the ground up.

---

## 5. Synthesis: What the Market Validated and Where the Gap Remains

| Capability | Upstart | Zest AI | Agentic APA (AA / AWS) | Halcyon Copilot |
|------------|---------|---------|------------------------|-----------------|
| Thin-file borrower coverage | ✅ Strong | ✅ Strong | Partial | ✅ Designed for |
| ML risk scoring | ✅ Proprietary | ✅ Lender-custom | Varies | ✅ XGBoost/LightGBM |
| SHAP / feature attribution | Partial | ✅ Full | Varies | ✅ Mandatory output |
| Fairness testing across segments | Partial | ✅ Full | Partial | ✅ Every eval run |
| Policy compliance via RAG | ❌ | ❌ | ❌ | ✅ ChromaDB + RAG |
| LLM-synthesised written reasons | ❌ | ❌ | Partial | ✅ Decision Synthesizer |
| Faithfulness evaluation loop | ❌ | ❌ | Partial | ✅ Evaluator-Optimizer |
| Stateful audit trail (replay) | Partial | ✅ Model only | ✅ Strong | ✅ Full state trace |
| Open-stack, self-hostable | ❌ | ❌ | ❌ | ✅ |
| Accessible to early-stage lender | ❌ | ❌ | ❌ | ✅ |

The market has proven that ML underwriting, fairness testing, multi-agent orchestration, and audit-trail persistence each work in production. No deployed system we found combines all of them in a single pipeline that an early-stage consumer lender can own, operate, and have reviewed by a regulator from first principles. That is the gap the Halcyon Credit Agentic Underwriting Copilot is designed to fill.

---

## 6. References

1. Upstart Holdings — *Our Story: Expanding Access to Affordable Credit* · upstart.com/our-story · Accessed June 2025
2. Upstart Holdings — *Q4 2024 Performance: 500+ Banks Adopt Upstart AI Lending Platform* · cobaltintelligence.com · February 2025
3. Upstart — *How AI Drives More Affordable Credit Access* · info.upstart.com · Accessed June 2025
4. Zest AI — *AI-Automated Credit Underwriting* · zest.ai · Accessed June 2025
5. Zest AI — *$200 Million Growth Investment from Insight Partners* · zest.ai · 2024
6. Zest AI — *Credit Models Proven to Increase Loan Approvals for Every Protected Class* · PR Newswire · January 31, 2024
7. Zest AI — *Why the CFPB Should Encourage AI in Underwriting* · zest.ai · September 2024
8. Automation Anywhere — *AI Agents for Loan Underwriting* · automationanywhere.com · June 2025
9. Automation Anywhere — *What is Agentic Process Automation? A Complete Guide* · automationanywhere.com · November 2025
10. Amazon Web Services — *Agentic AI in Financial Services: Choosing the Right Pattern for Multi-Agent Systems* · aws.amazon.com · December 2025
11. Datamatics — *Agentic AI Underwriting Case Study: 80% Faster Insurance Decisions* · datamatics.com · Accessed June 2025
12. TIMVERO — *How AI Is Transforming Lending in 2026* · timvero.com · May 2026
13. Mohammad Asif Ali — *Efficient Underwriting Using Agentic AI* · Software Engineering, Vol. 12 No. 1, 2025 · doi:10.5923/j.se.20251201.01
14. FinRegLab — *ML Models in Credit Underwriting: Independent Evaluation* · July 2025 (cited in buildmvpfast.com)
15. McKinsey & Company — *AI-driven credit models: 10,000 data points per borrower* · 2024 (cited in timvero.com)

---

*This document is part of the Futurense AI Clinic Capstone Program academic portfolio. Halcyon Credit is a fictional persona.*
